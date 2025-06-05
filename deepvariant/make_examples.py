# Copyright 2017 Google LLC.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Step one of DeepVariant: creates tf.Example protos for training/calling."""

import itertools
import os
import random
import socket
import tensorflow as tf
import timeit
import zlib

from absl import app
from absl import flags
from pathlib import Path
from typing import List

import dataclasses
import ray

from deepvariant import dv_constants
from deepvariant import logging_level
from deepvariant import make_examples_core
from deepvariant import make_examples_options
from deepvariant.protos import deepvariant_pb2
from ray.runtime_env import RuntimeEnv
from third_party.nucleus.io.python import hts_verbose
from third_party.nucleus.util import errors
from third_party.nucleus.util import proto_utils
from typing import List, TypeVar, Sequence, Tuple

# Sentinel command line flag value indicating no downsampling should occur.
NO_DOWNSAMPLING = 0.0

MAIN_SAMPLE_INDEX = 0  # 0 is the only sample.

FLAGS = flags.FLAGS

# Adopt more general flags from make_examples_options.
flags.adopt_module_key_flags(make_examples_options)

# Flags related to the sample in DeepVariant:
READS_ = flags.DEFINE_string(
    'reads',
    None,
    (
        'Required. Aligned, sorted, indexed BAM file containing the reads we'
        ' want to call. Should be aligned to a reference genome compatible with'
        ' --ref. Can provide multiple BAMs (comma-separated).'
    ),
)
SAMPLE_NAME_ = flags.DEFINE_string(
    'sample_name',
    '',
    (
        'Sample name to use for our sample_name in the output'
        ' Variant/DeepVariantCall protos. If not specified, will be inferred'
        ' from the header information from --reads.'
    ),
)
PILEUP_IMAGE_HEIGHT_ = flags.DEFINE_integer(
    'pileup_image_height',
    0,
    'Height for the pileup image. If 0, uses the default height',
)
DOWNSAMPLE_FRACTION_ = flags.DEFINE_float(
    'downsample_fraction',
    NO_DOWNSAMPLING,
    (
        f'If not {NO_DOWNSAMPLING} must be a value between 0.0 and 1.0. Reads'
        ' will be kept (randomly) with a probability of downsample_fraction'
        ' from the input BAM. This argument makes it easy to create examples'
        ' as though the input BAM had less coverage.'
    ),
)
PROPOSED_VARIANTS_ = flags.DEFINE_string(
    'proposed_variants',
    '',
    (
        '(Only used when --variant_caller=vcf_candidate_importer.) '
        'Tabix-indexed VCF file containing the proposed positions and alts for '
        '`vcf_candidate_importer`. The GTs will be ignored.'
    ),
)
CANDIDATE_POSITIONS_ = flags.DEFINE_string(
    'candidate_positions',
    None,
    'Path to the binary file containing candidate positions.',
)
N_WORKERS = flags.DEFINE_integer(
    'num_workers',
    12,
    'Number of ray workers',
)
DEBUG_VERSION = flags.DEFINE_integer(
    'debug_version',
    0,
    'Whether to use main, main_ray or main_ray2',
)

def one_sample_from_flags(add_flags=True, flags_obj=None):
  """Collect sample-related options into a list of samples."""
  # Sample-specific options.
  sample_name = make_examples_core.assign_sample_name(
      sample_name_flag=SAMPLE_NAME_.value, reads_filenames=READS_.value
  )
  sample_options = deepvariant_pb2.SampleOptions(
      role='main_sample',
      name=sample_name,
      variant_caller_options=make_examples_core.make_vc_options(
          sample_name=sample_name, flags_obj=flags_obj
      ),
      order=[0],
      pileup_height=dv_constants.PILEUP_DEFAULT_HEIGHT,
  )

  if add_flags:
    if READS_.value:
      sample_options.reads_filenames.extend(READS_.value.split(','))
    if DOWNSAMPLE_FRACTION_.value != NO_DOWNSAMPLING:
      sample_options.downsample_fraction = DOWNSAMPLE_FRACTION_.value
    if PILEUP_IMAGE_HEIGHT_.value:
      sample_options.pileup_height = PILEUP_IMAGE_HEIGHT_.value
    if PROPOSED_VARIANTS_.value:
      sample_options.proposed_variants_filename = PROPOSED_VARIANTS_.value
    if CANDIDATE_POSITIONS_.value:
      sample_options.candidate_positions = CANDIDATE_POSITIONS_.value
  samples_in_order = [sample_options]
  sample_role_to_train = sample_options.role
  return samples_in_order, sample_role_to_train


def default_options(add_flags=True, flags_obj=None):
  """Creates a MakeExamplesOptions proto populated with reasonable defaults.

  Args:
    add_flags: bool. defaults to True. If True, we will push the value of
      certain FLAGS into our options. If False, those option fields are left
      uninitialized.
    flags_obj: object.  If not None, use as the source of flags, else use global
      FLAGS.

  Returns:
    deepvariant_pb2.MakeExamplesOptions protobuf.

  Raises:
    ValueError: If we observe invalid flag values.
  """
  if not flags_obj:
    flags_obj = FLAGS

  samples_in_order, sample_role_to_train = one_sample_from_flags(
      add_flags=add_flags, flags_obj=flags_obj
  )

  options = make_examples_options.shared_flags_to_options(
      add_flags=add_flags,
      flags_obj=flags_obj,
      samples_in_order=samples_in_order,
      sample_role_to_train=sample_role_to_train,
      main_sample_index=MAIN_SAMPLE_INDEX,
  )

  if add_flags:
    options.bam_fname = os.path.basename(READS_.value)
  return options


def check_options_are_valid(options):
  """Checks that all the options chosen make sense together."""

  # Check for general flags (shared for DeepVariant and DeepTrio).
  make_examples_options.check_options_are_valid(
      options, main_sample_index=MAIN_SAMPLE_INDEX
  )

  main_sample = options.sample_options[MAIN_SAMPLE_INDEX]
  if (
      options.mode == deepvariant_pb2.MakeExamplesOptions.CANDIDATE_SWEEP
      and main_sample.candidate_positions is None
  ):
    errors.log_and_raise(
        '--candidate_positions is required when --positions_sweep is set.'
    )
  if (
      options.mode == deepvariant_pb2.MakeExamplesOptions.CANDIDATE_SWEEP
      and main_sample.proposed_variants_filename
  ):
    errors.log_and_raise(
        '--positions_sweep_mode is incompatible with --proposed_variants'
    )
  if main_sample.candidate_positions and main_sample.proposed_variants_filename:
    errors.log_and_raise(
        '--candidate_positions is incompatible with --proposed_variants'
    )

def main_ray(argv=()):
    import os
    import socket
    import sys

    driver_pythonpath = os.environ.get("PYTHONPATH")
    ld_lib_path = os.environ.get("LD_LIBRARY_PATH")
    pid = os.getpid()
    print(f"{pid} ld_lib_path: {ld_lib_path}")
    hostname = socket.gethostname()
    pid = os.getpid()
    sys.path.append('/opt/deepvariant/unzipped/runfiles/com_google_deepvariant')
    print(f"The process ID is: {pid}")
    print(f"{pid} hostname: {hostname}")
    print(f"{pid} Propagating PYTHONPATH to Ray workers: {driver_pythonpath}")
    print(f"{pid} sys.path: {sys.path}")

    num_buckets: int = N_WORKERS.value
    batch_size: int = 256
    file_prefix = "train"
    # Set up options in the main process
    options = default_options(add_flags=True, flags_obj=FLAGS)
    check_options_are_valid(options)

    output_dir: Path = os.path.dirname(options.examples_filename)

    # Serialize the options protobuf
    options_serialized = options.SerializeToString()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting image processing system data={output_dir=}")

    ray.init()

    t0 = timeit.default_timer()

    # Create writer actors (one per node)
    num_writers = N_WORKERS.value
    print(f'Starting {num_writers} writer actors')
    bucket = Bucket(num_buckets, num_writers)
    writer_actors = [
        TFRecordWriter.remote(i, bucket[i], batch_size, output_dir, file_prefix) for i in range(num_writers)
    ]

    # Create reader actors (one per node)
    num_make_examples_proc = N_WORKERS.value
    print(f'Starting {num_make_examples_proc} reader actors')
    make_example_procs = []
    for shard_id in range(num_make_examples_proc):
      make_example_procs.append(ShuffleAndStream.remote(shard_id, bucket, writer_actors, batch_size, num_make_examples_proc, options_serialized))

    make_examples_tasks = [reader.create_and_stream_examples.remote()
        for reader in make_example_procs
    ]

    # Wait for all generation tasks to complete
    read_count = ray.get(make_examples_tasks)
    total_reads = sum(read_count)
    print(f"Total record processed: {total_reads}")

    # Flush all writers to ensure all records are written
    writer_tasks = [writer.save_records.remote() for writer in writer_actors]
    written_counts = ray.get(writer_tasks)
    total_written = sum(written_counts)
    print(f"Total written records: {total_written}")

    assert total_reads == total_written, "Mismatch between generated and written records!"
    print("Processing completed successfully")

    print(f"Total time taken: {timeit.default_timer() - t0:.2f} seconds")


@ray.remote
class ShuffleAndStream:
    """Ray actor for reading TFRecord files."""
    def __init__(self, node_id: int, bucket: "Bucket", writer_actors: List, batch_size: int = 1, num_shards=None, options_serialized=None):
        import socket
        self.node_id = node_id
        self.hostname = socket.gethostname()
        self.writer_actors = writer_actors
        self.num_writers = len(writer_actors)

        self.bucket = bucket
        self.batch_size = batch_size
        self.buffer = ReadBuffer(self.node_id, bucket.num_buckets * batch_size)

        self.sent_count = 0
        self.sent_events = 0
        self.read_count = 0

        self.shard_id = node_id
        self.num_shards = num_shards
        self.options_serialized = options_serialized

        import os
        import socket
        import sys

        driver_pythonpath = os.environ.get("PYTHONPATH")
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH")
        pid = os.getpid()
        print(f"{pid} ld_lib_path: {ld_lib_path}")
        hostname = socket.gethostname()

        sys.path.append('/opt/deepvariant/unzipped/runfiles/com_google_deepvariant')
        print(f"The process ID is: {pid}")
        print(f"{pid} hostname: {hostname}")
        print(f"{pid} Propagating PYTHONPATH to Ray workers: {driver_pythonpath}")
        print(f"{pid} sys.path: {sys.path}")

    def batch_shuffle_and_send(self) -> None:
        """Shuffle and send a batch of records to random writers."""
        records = self.buffer.shuffle()
        for i in range(self.num_writers):
            s, e, _ = self.bucket[i]
            s, e = s * self.batch_size, e * self.batch_size
            print(f"Reader {self.shard_id} Sending to Writer {i}")
            ray.get(self.writer_actors[i].stream_record.remote(records[s:e]))

        self.sent_count += sum(1 for record in records if record is not None)
        self.buffer.reset_buffer()

        self.sent_events += 1
        if self.sent_events % 100 == 0:
            print(f"Reader {self.node_id} has sent {self.sent_count} records")

    def create_and_stream_examples(self):
        from deepvariant import make_examples_core, logging_level
        from deepvariant.protos import deepvariant_pb2
        from third_party.nucleus.io.python import hts_verbose

        # Setup minimal flags environment for the worker
        from absl import flags
        FLAGS = flags.FLAGS
        # Initialize flags without parsing command line
        try:
            FLAGS.mark_as_parsed()
        except:
            # In case flags were already parsed
            pass

        # Set HTS logging level
        logging_level.set_from_flag()
        hts_verbose.set(hts_verbose.htsLogLevel.HTS_LOG_ERROR)
        print(f"Starting shard {self.shard_id}")

        # Deserialize options
        options = deepvariant_pb2.MakeExamplesOptions()
        options.ParseFromString(self.options_serialized)

        # Set shard-specific options
        # this examples_filename is not used, keeping here for now
        options.examples_filename = f'{options.examples_filename}-{self.shard_id:05d}-of-{self.num_shards:05d}'
        options.task_id = self.shard_id  # this is how each process know which chunk of bam to work on
        options.num_shards = self.num_shards

        def callback(example_bytes: bytes):
            example_bytes = zlib.compress(example_bytes)
            if example_bytes is None:
                print("Received None from make examples")
            self.buffer.add_record(example_bytes)
            if self.buffer.is_full():
                print(f"Buffer is full with ")
                self.batch_shuffle_and_send()

        make_examples_core.make_examples_runner(options, callback)
        self.batch_shuffle_and_send()
        return self.sent_count


@ray.remote
class TFRecordWriter:
    """
    Actor that receives TF records and writes them to local files.
    Each node has one ImageWriter instance.
    """
    def __init__(self, node_id: int, partition,  batch_size: int = 0, output_dir: str = "/tmp", file_prefix="images"):
        import socket
        self.node_id = node_id
        self.output_dir = output_dir
        self.partition = partition
        self.file_prefix = file_prefix
        self.hostname = socket.gethostname()
        self.current_records = []
        self.batch_size = batch_size
        self.total_records_received = 0
        self.total_records_written = 0
        self._tfrecord_writer = None
        self.files = [self.get_writer(filename) for filename in self.filenames]
        print(f"ShuffleWriter {node_id} initialized on {self.hostname}")
        import os

        import sys

        driver_pythonpath = os.environ.get("PYTHONPATH")
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH")
        pid = os.getpid()
        print(f"{pid} ld_lib_path: {ld_lib_path}")
        hostname = socket.gethostname()

        sys.path.append('/opt/deepvariant/unzipped/runfiles/com_google_deepvariant')
        print(f"The process ID is: {pid}")
        print(f"{pid} hostname: {hostname}")
        print(f"{pid} Propagating PYTHONPATH to Ray workers: {driver_pythonpath}")
        print(f"{pid} sys.path: {sys.path}")

    @property
    def filenames(self) -> list[str]:
        """Format string for TFRecord file naming"""
        start, end, n = self.partition
        return [f"{self.file_prefix}.tfrecord-{i:05d}-of-{n:05d}.gz" for i in range(start, end)]

    def get_writer(self, filename) -> tf.io.TFRecordWriter:
        # Create the TFRecord writer
        filename = os.path.join(self.output_dir, filename)
        return tf.io.TFRecordWriter(filename, options=tf.io.TFRecordOptions(compression_type="GZIP"))

    def stream_record(self, compressed_tf_records: list[bytes]) -> bool:
        """Receive a TF record from any generator and write to local file"""
        def batched(iterable, n):
            """Batch data into lists of length n. The last batch may be shorter."""
            # batched('ABCDEFG', 3) --> ABC DEF G
            it = iter(iterable)
            while batch := list(itertools.islice(it, n)):
                yield batch
        none_count = 0
        for i, batched_recs in enumerate(batched(compressed_tf_records, self.batch_size)):
            for compressed_tf_record in batched_recs:
                if compressed_tf_record is None:
                    none_count += 1
                    continue
                record = zlib.decompress(compressed_tf_record)
                self.files[i].write(record)

                # Log progress periodically
                self.total_records_written += 1
                if self.total_records_written % 1000 == 0:
                    print(f"Writer {self.node_id} on {self.hostname} wrote {self.total_records_written} records")
        total_records = len(compressed_tf_records)  # should be = batch size
        print(f"Writer {self.node_id} received {total_records - none_count} out of {total_records}")
        return True

    def save_records(self) -> int:
        """Writes examples to a TFRecord file.
        Args:
            examples: List of serialized tf.Examples.
            output_path: Base path for output files.
            shard_index: Index of this shard.
            num_shards: Total number of shards.
        Returns:
            Number of examples written.
        """
        # Use TensorFlow 2 API to write records
        for writer in self.files:
            writer.close()
        self._tfrecord_writer = None
        self.current_records = []
        return self.total_records_written


def tf_records(filename: Path):
    """
    Generator function to read TFRecord files.
    Args:
        filename: Path to the TFRecord file.
    Yields:
        Serialized tf.Example records.
    """
    compression_type = 'GZIP' if filename.suffix == '.gz' else ''
    dataset = tf.data.TFRecordDataset(filename, compression_type=compression_type)
    for raw_record in dataset:
        raw_data = raw_record.numpy()
        compressed_data = zlib.compress(raw_data)
        yield compressed_data


def get_balanced_partition_indices(n: int, num_partitions: int) -> List[Tuple[int, int]]:
    """
    Generate start and end indices for balanced partitions of a list of length n.

    Args:
        n: The length of the list to partition
        num_partitions: The number of partitions to create

    Returns:
        A list of tuples (start_idx, end_idx) for each partition,
        where each partition includes elements from start_idx (inclusive)
        to end_idx (exclusive)
    """
    # Handle edge cases
    if num_partitions <= 0:
        raise ValueError("Number of partitions must be positive")

    # If we have fewer elements than partitions, some partitions will be empty
    if n < num_partitions:
        result = []
        for i in range(num_partitions):
            if i < n:
                result.append((i, i + 1))  # Each element gets its own partition
            else:
                result.append((n, n))  # Empty partition
        return result

    # Calculate the size of each partition
    # Some partitions will have base_size elements, others will have base_size + 1
    base_size = n // num_partitions
    remainder = n % num_partitions

    result = []
    start_idx = 0

    for i in range(num_partitions):
        # The first 'remainder' partitions get an extra element
        partition_size = base_size + (1 if i < remainder else 0)
        end_idx = start_idx + partition_size

        # Store the start and end indices
        result.append((start_idx, end_idx))

        # Update the start index for the next partition
        start_idx = end_idx

    return result



@dataclasses.dataclass
class Bucket:
    num_buckets: int
    num_writers: int
    def __post_init__(self):
        self.partitions = get_balanced_partition_indices(self.num_buckets, self.num_writers)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.partitions[item] + (self.num_buckets,)
        else:
            raise TypeError("Invalid index type")

    def num_records_for_partition(self, i: int):
        """Get the number of records for a given partition"""
        start, end, n = self.partitions[i]
        return end - start


class ReadBuffer:
    def __init__(self, reader_id: int, buffer_size: int):
        self.reader_id = reader_id
        self.size = buffer_size
        self.records: list[bytes] = [None] * buffer_size
        self.index = 0
        self.num_reads = 0

    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return self.index >= self.size

    def add_record(self, record: bytes) -> None:
        """Add a record to the buffer."""
        self.records[self.index] = record
        self.index += 1
        self.num_reads += 1
        if self.num_reads % 1000 == 0:
            print(f"Reader {self.reader_id} has read {self.num_reads} records")

    def shuffle(self):
        """Shuffle the records in the buffer."""
        random.shuffle(self.records)
        return self.records

    def reset_buffer(self) -> None:
        """Reset the buffer."""
        self.index = 0


if __name__ == '__main__':
    flags.mark_flags_as_required([
      'examples',
      'mode',
      'reads',
      'ref',
    ])
    app.run(main_ray)
