"""
A :class:`~allennlp.data.fields.field.Field` is some piece of data instance
that ends up as an array in a model.
"""

from radgraph.allennlp.data.fields.field import Field
from radgraph.allennlp.data.fields.adjacency_field import AdjacencyField
from radgraph.allennlp.data.fields.array_field import ArrayField
from radgraph.allennlp.data.fields.flag_field import FlagField
from radgraph.allennlp.data.fields.index_field import IndexField
from radgraph.allennlp.data.fields.label_field import LabelField
from radgraph.allennlp.data.fields.list_field import ListField
from radgraph.allennlp.data.fields.metadata_field import MetadataField
from radgraph.allennlp.data.fields.multilabel_field import MultiLabelField
from radgraph.allennlp.data.fields.namespace_swapping_field import NamespaceSwappingField
from radgraph.allennlp.data.fields.sequence_field import SequenceField
from radgraph.allennlp.data.fields.sequence_label_field import SequenceLabelField
from radgraph.allennlp.data.fields.span_field import SpanField
from radgraph.allennlp.data.fields.text_field import TextField
