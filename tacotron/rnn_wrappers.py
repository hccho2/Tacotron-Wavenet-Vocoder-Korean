# coding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import rnn_cell_impl
#from tensorflow.contrib.data.python.util import nest
from tensorflow.contrib.framework import nest
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _bahdanau_score, _BaseAttentionMechanism, BahdanauAttention, \
                             AttentionWrapperState, AttentionMechanism, _BaseMonotonicAttentionMechanism,_maybe_mask_score,_prepare_memory,_monotonic_probability_fn

from tensorflow.python.layers.core import Dense
from .modules import prenet
import functools
_zero_state_tensors = rnn_cell_impl._zero_state_tensors



class AttentionWrapper(RNNCell):
    """Wraps another `RNNCell` with attention.
    """

    def __init__(self,
                 cell,
                 attention_mechanism,
                 is_manual_attention,   # 추가된 argument
                 manual_alignments,     # 추가된 argument
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None):
        """Construct the `AttentionWrapper`.
        **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
        `AttentionWrapper`, then you must ensure that:
        - The encoder output has been tiled to `beam_width` via
          @{tf.contrib.seq2seq.tile_batch} (NOT `tf.tile`).
        - The `batch_size` argument passed to the `zero_state` method of this
          wrapper is equal to `true_batch_size * beam_width`.
        - The initial state created with `zero_state` above contains a
          `cell_state` value containing properly tiled final state from the
          encoder.
        An example:
        ```
        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
            encoder_outputs, multiplier=beam_width)
        tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
            encoder_final_state, multiplier=beam_width)
        tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
            sequence_length, multiplier=beam_width)
        attention_mechanism = MyFavoriteAttentionMechanism(
            num_units=attention_depth,
            memory=tiled_inputs,
            memory_sequence_length=tiled_sequence_length)
        attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
        decoder_initial_state = attention_cell.zero_state(
            dtype, batch_size=true_batch_size * beam_width)
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=tiled_encoder_final_state)
        ```
        Args:
          cell: An instance of `RNNCell`.
          attention_mechanism: A list of `AttentionMechanism` instances or a single
            instance.
          attention_layer_size: A list of Python integers or a single Python
            integer, the depth of the attention (output) layer(s). If None
            (default), use the context as attention at each time step. Otherwise,
            feed the context and cell output into the attention layer to generate
            attention at each time step. If attention_mechanism is a list,
            attention_layer_size must be a list of the same length.
          alignment_history: Python boolean, whether to store alignment history
            from all time steps in the final output state (currently stored as a
            time major `TensorArray` on which you must call `stack()`).
          cell_input_fn: (optional) A `callable`.  The default is:
            `lambda inputs, attention: tf.concat([inputs, attention], -1)`.
          output_attention: Python bool.  If `True` (default), the output at each
            time step is the attention value.  This is the behavior of Luong-style
            attention mechanisms.  If `False`, the output at each time step is
            the output of `cell`.  This is the behavior of Bhadanau-style
            attention mechanisms.  In both cases, the `attention` tensor is
            propagated to the next time step via the state and is used there.
            This flag only controls whether the attention mechanism is propagated
            up to the next cell in an RNN stack or to the top RNN output.
          initial_cell_state: The initial state value to use for the cell when
            the user calls `zero_state()`.  Note that if this value is provided
            now, and the user uses a `batch_size` argument of `zero_state` which
            does not match the batch size of `initial_cell_state`, proper
            behavior is not guaranteed.
          name: Name to use when creating ops.
        Raises:
          TypeError: `attention_layer_size` is not None and (`attention_mechanism`
            is a list but `attention_layer_size` is not; or vice versa).
          ValueError: if `attention_layer_size` is not None, `attention_mechanism`
            is a list, and its length does not match that of `attention_layer_size`.
        """
        super(AttentionWrapper, self).__init__(name=name)
        
        self.is_manual_attention = is_manual_attention
        self.manual_alignments = manual_alignments

        
        rnn_cell_impl.assert_like_rnncell("cell", cell)
        if isinstance(attention_mechanism, (list, tuple)):
            self._is_multi = True
            attention_mechanisms = attention_mechanism
            for attention_mechanism in attention_mechanisms:
                if not isinstance(attention_mechanism, AttentionMechanism):
                    raise TypeError(
                        "attention_mechanism must contain only instances of "
                        "AttentionMechanism, saw type: %s"
                        % type(attention_mechanism).__name__)
        else:
            self._is_multi = False
            if not isinstance(attention_mechanism, AttentionMechanism):
                raise TypeError(
                    "attention_mechanism must be an AttentionMechanism or list of "
                    "multiple AttentionMechanism instances, saw type: %s"
                    % type(attention_mechanism).__name__)
            attention_mechanisms = (attention_mechanism,)

        if cell_input_fn is None:
            cell_input_fn = (
                lambda inputs, attention: tf.concat([inputs, attention], -1))
        else:
            if not callable(cell_input_fn):
                raise TypeError(
                    "cell_input_fn must be callable, saw type: %s"
                    % type(cell_input_fn).__name__)

        if attention_layer_size is not None:
            attention_layer_sizes = tuple(
                attention_layer_size
                if isinstance(attention_layer_size, (list, tuple))
                else (attention_layer_size,))
            if len(attention_layer_sizes) != len(attention_mechanisms):
                raise ValueError(
                    "If provided, attention_layer_size must contain exactly one "
                    "integer per attention_mechanism, saw: %d vs %d"
                    % (len(attention_layer_sizes), len(attention_mechanisms)))
            self._attention_layers = tuple(
                layers_core.Dense(
                    attention_layer_size,
                    name="attention_layer",
                    use_bias=False,
                    dtype=attention_mechanisms[i].dtype)
                for i, attention_layer_size in enumerate(attention_layer_sizes))
            self._attention_layer_size = sum(attention_layer_sizes)
        else:
            self._attention_layers = None
            self._attention_layer_size = sum(
                attention_mechanism.values.get_shape()[-1].value
                for attention_mechanism in attention_mechanisms)

        self._cell = cell
        self._attention_mechanisms = attention_mechanisms
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._alignment_history = alignment_history
        with tf.name_scope(name, "AttentionWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (
                    final_state_tensor.shape[0].value
                    or tf.shape(final_state_tensor)[0])
                error_message = (
                    "When constructing AttentionWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and initial_cell_state.  Are you using "
                    "the BeamSearchDecoder?  You may need to tile your initial state "
                    "via the tf.contrib.seq2seq.tile_batch function with argument "
                    "multiple=beam_width.")
                with tf.control_dependencies(
                    self._batch_size_checks(state_batch_size, error_message)):
                    self._initial_cell_state = nest.map_structure(
                        lambda s: tf.identity(s, name="check_initial_cell_state"),
                        initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        return [tf.assert_equal(batch_size,
                                       attention_mechanism.batch_size,
                                       message=error_message)
                for attention_mechanism in self._attention_mechanisms]

    def _item_or_tuple(self, seq):
        """Returns `seq` as tuple or the singular element.
        Which is returned is determined by how the AttentionMechanism(s) were passed
        to the constructor.
        Args:
          seq: A non-empty sequence of items or generator.
        Returns:
           Either the values in the sequence as a tuple if AttentionMechanism(s)
           were passed to the constructor as a sequence or the singular element.
        """
        t = tuple(seq)
        if self._is_multi:
            return t
        else:
            return t[0]

    @property
    def output_size(self):
        if self._output_attention:
            return self._attention_layer_size
        else:
            return self._cell.output_size

    @property
    def state_size(self):
        """The `state_size` property of `AttentionWrapper`.
        Returns:
          An `AttentionWrapperState` tuple containing shapes used by this object.
        """
        return AttentionWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            attention=self._attention_layer_size,
            alignments=self._item_or_tuple(
                a.alignments_size for a in self._attention_mechanisms),
            attention_state=self._item_or_tuple(
                a.state_size for a in self._attention_mechanisms),
            alignment_history=self._item_or_tuple(
                a.alignments_size if self._alignment_history else ()
                for a in self._attention_mechanisms))  # sometimes a TensorArray

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.
        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.
        Args:
          batch_size: `0D` integer tensor: the batch size.
          dtype: The internal state data type.
        Returns:
          An `AttentionWrapperState` tuple containing zeroed out tensors and,
          possibly, empty `TensorArray` objects.
        Raises:
          ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                "When calling zero_state of AttentionWrapper %s: " % self._base_name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the requested batch size.  Are you using "
                "the BeamSearchDecoder?  If so, make sure your encoder output has "
                "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                "the batch_size= argument passed to zero_state is "
                "batch_size * beam_width.")
            with tf.control_dependencies(
                self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: tf.identity(s, name="checked_cell_state"),
                    cell_state)
            initial_alignments = [
                attention_mechanism.initial_alignments(batch_size, dtype)
                for attention_mechanism in self._attention_mechanisms]
            return AttentionWrapperState(
                cell_state=cell_state,
                time=tf.zeros([], dtype=tf.int32),
                attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                              dtype),
                alignments=self._item_or_tuple(initial_alignments),
                attention_state=self._item_or_tuple(
                    attention_mechanism.initial_state(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms),
                alignment_history=self._item_or_tuple(
                    tf.TensorArray(
                        dtype,
                        size=0,
                        dynamic_size=True,
                        element_shape=alignment.shape)
                    if self._alignment_history else ()
                    for alignment in initial_alignments))

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).
        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:
          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.
        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead."  % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)  # concat
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with tf.control_dependencies(
            self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = tf.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_attention_state = _compute_attention(
                attention_mechanism, cell_output, previous_attention_state[i],
                self._attention_layers[i] if self._attention_layers else None,
                self.is_manual_attention, self.manual_alignments,state.time)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = tf.concat(all_attentions, 1)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state

def _compute_attention(attention_mechanism, cell_output, previous_alignments,attention_layer, is_manual_attention, manual_alignments, time):

    computed_alignments, next_attention_state  = attention_mechanism(cell_output, state=previous_alignments) # (query, state)를 넘긴다.
    batch_size, max_time = tf.shape(computed_alignments)[0], tf.shape(computed_alignments)[1]

    alignments = tf.cond(is_manual_attention, lambda: manual_alignments[:, time, :],lambda: computed_alignments,)    # 여기 이곳만 tensorflow 1.3과 다름.

    #alignments = tf.one_hot(tf.zeros((batch_size,), dtype=tf.int32), max_time, dtype=tf.float32)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)

    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #         [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #         [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #         [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = tf.matmul(expanded_alignments, attention_mechanism.values)
    context = tf.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(tf.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments,next_attention_state


class DecoderPrenetWrapper(RNNCell):
    '''Runs RNN inputs through a prenet before sending them to the cell.'''
    #  input에 prenet을 먼저 적용하는 것 뿐이다.
    def __init__(self, cell, embed_to_concat,is_training, prenet_sizes, dropout_prob):

        super(DecoderPrenetWrapper, self).__init__()
        self._is_training = is_training

        self._cell = cell
        self._embed_to_concat = embed_to_concat

        self.prenet_sizes = prenet_sizes
        self.dropout_prob = dropout_prob

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def call(self, inputs, state):
        prenet_out = prenet(inputs, self._is_training,self.prenet_sizes, self.dropout_prob, scope='decoder_prenet')

        if self._embed_to_concat is not None:
            concat_out = tf.concat([prenet_out, self._embed_to_concat],axis=-1, name='speaker_concat')
            return self._cell(concat_out, state)
        else:
            return self._cell(prenet_out, state)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)



class ConcatOutputAndAttentionWrapper(RNNCell):
    '''Concatenates RNN cell output with the attention context vector.

    This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
    attention_layer_size=None and output_attention=False. Such a cell's state will include an
    "attention" field that is the context vector.
    '''
    def __init__(self, cell, embed_to_concat):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell
        self._embed_to_concat = embed_to_concat

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)

        if self._embed_to_concat is not None:
            tensors = [output, res_state.attention,self._embed_to_concat,]
            return tf.concat(tensors, axis=-1), res_state
        else:
            return tf.concat([output, res_state.attention], axis=-1), res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)



class BahdanauMonotonicAttention_hccho(_BaseMonotonicAttentionMechanism):
    """Monotonic attention mechanism with Bahadanau-style energy function.

    This type of attention enforces a monotonic constraint on the attention
    distributions; that is once the model attends to a given point in the memory
    it can't attend to any prior points at subsequence output timesteps.  It
    achieves this by using the _monotonic_probability_fn instead of softmax to
    construct its attention distributions.  Since the attention scores are passed
    through a sigmoid, a learnable scalar bias parameter is applied after the
    score function and before the sigmoid.  Otherwise, it is equivalent to
    BahdanauAttention.  This approach is proposed in

    Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
    "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
    ICML 2017.  https://arxiv.org/abs/1704.00784
    """

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 score_mask_value=None,
                 sigmoid_noise=0.,
                 sigmoid_noise_seed=None,
                 score_bias_init=0.,
                 mode="parallel",
                 dtype=None,
                 name="BahdanauMonotonicAttentionHccho"):
        """Construct the Attention mechanism.

        Args:
          num_units: The depth of the query mechanism.
          memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
          normalize: Python boolean.  Whether to normalize the energy term.
          score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
          sigmoid_noise: Standard deviation of pre-sigmoid noise.  See the docstring
            for `_monotonic_probability_fn` for more information.
          sigmoid_noise_seed: (optional) Random seed for pre-sigmoid noise.
          score_bias_init: Initial value for score bias scalar.  It's recommended to
            initialize this to a negative value when the length of the memory is
            large.
          mode: How to compute the attention distribution.  Must be one of
            'recursive', 'parallel', or 'hard'.  See the docstring for
            `tf.contrib.seq2seq.monotonic_attention` for more information.
          dtype: The data type for the query and memory layers of the attention
            mechanism.
          name: Name to use when creating ops.
        """
        # Set up the monotonic probability fn with supplied parameters
        if dtype is None:
            dtype = tf.float32
        wrapped_probability_fn = functools.partial(
            _monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode,
            seed=sigmoid_noise_seed)
        super(BahdanauMonotonicAttention_hccho, self).__init__(
            query_layer=Dense(num_units, name="query_layer", use_bias=False, dtype=dtype),
            memory_layer=Dense(num_units, name="memory_layer", use_bias=False, dtype=dtype),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name
        self._score_bias_init = score_bias_init

    def __call__(self, query, state):
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with tf.variable_scope(None, "bahdanau_monotonic_hccho_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _bahdanau_score(processed_query, self._keys, self._normalize)     # keys 가 memory임
            score_bias = tf.get_variable("attention_score_bias", dtype=processed_query.dtype, initializer=self._score_bias_init)

            #alignments_bias = tf.get_variable("alignments_bias", shape = state.get_shape()[-1],dtype=processed_query.dtype, initializer=tf.zeros_initializer())  # hccho
            alignments_bias = tf.get_variable("alignments_bias", shape = (1),dtype=processed_query.dtype, initializer=tf.zeros_initializer())  # hccho

            score += score_bias
        alignments = self._probability_fn(score, state)   #BahdanauAttention에서 _probability_fn = softmax

        next_state = alignments   # 다음 alignment 계산에 사용할 state 값  =  AttentionWrapperState.attention_state
        # hccho. alignment가 attention 계산에 직접 사용된다.
        alignments = tf.nn.relu(alignments+alignments_bias)
        alignments = alignments/(tf.reduce_sum(alignments,axis=-1,keepdims=True) + 1.0e-12 )  # hccho 수정


        return alignments, next_state



class LocationSensitiveAttention(BahdanauAttention):
    """Impelements Bahdanau-style (cumulative) scoring function.
    Usually referred to as "hybrid" attention (content-based + location-based)
    Extends the additive attention described in:
    "D. Bahdanau, K. Cho, and Y. Bengio, �쏯eural machine transla-
tion by jointly learning to align and translate,�� in Proceedings
of ICLR, 2015."
    to use previous alignments as additional location features.

    This attention is described in:
    J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
gio, �쏛ttention-based models for speech recognition,�� in Ad-
vances in Neural Information Processing Systems, 2015, pp.
577��585.
    """

    def __init__(self,
                    num_units,
                    memory,
                    memory_sequence_length=None,
                    smoothing=False,
                    cumulate_weights=True,
                    name='LocationSensitiveAttention'):
        """Construct the Attention mechanism.
        Args:
                num_units: The depth of the query mechanism.
                memory: The memory to query; usually the output of an RNN encoder.  This
                        tensor should be shaped `[batch_size, max_time, ...]`.
                memory_sequence_length (optional): Sequence lengths for the batch entries
                        in memory.  If provided, the memory tensor rows are masked with zeros
                        for values past the respective sequence lengths. Only relevant if mask_encoder = True.
                smoothing (optional): Boolean. Determines which normalization function to use.
                        Default normalization function (probablity_fn) is softmax. If smoothing is
                        enabled, we replace softmax with:
                                        a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
                        Introduced in:
                                J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
                          gio, �쏛ttention-based models for speech recognition,�� in Ad-
                          vances in Neural Information Processing Systems, 2015, pp.
                          577��585.
                        This is mainly used if the model wants to attend to multiple inputs parts
                        at the same decoding step. We probably won't be using it since multiple sound
                        frames may depend from the same character, probably not the way around.
                        Note:
                                We still keep it implemented in case we want to test it. They used it in the
                                paper in the context of speech recognition, where one phoneme may depend on
                                multiple subsequent sound frames.
                name: Name to use when creating ops.
        """
        #Create normalization function
        #Setting it to None defaults in using softmax
        normalization_function = _smoothing_normalization if (smoothing == True) else None
        super(LocationSensitiveAttention, self).__init__(
                        num_units=num_units,
                        memory=memory,
                        memory_sequence_length=memory_sequence_length,
                        probability_fn=normalization_function,
                        name=name)

        self.location_convolution = tf.layers.Conv1D(filters=32,
                kernel_size=(31, ), padding='same', use_bias=True,
                bias_initializer=tf.zeros_initializer(), name='location_features_convolution')
        self.location_layer = tf.layers.Dense(units=num_units, use_bias=False,
                dtype=tf.float32, name='location_features_layer')
        self._cumulate = cumulate_weights

    def __call__(self, query, state):
        """Score the query based on the keys and values.
        Args:
                query: Tensor of dtype matching `self.values` and shape
                        `[batch_size, query_depth]`.
                state (previous alignments): Tensor of dtype matching `self.values` and shape
                        `[batch_size, alignments_size]`
                        (`alignments_size` is memory's `max_time`).
        Returns:
                alignments: Tensor of dtype matching `self.values` and shape
                        `[batch_size, alignments_size]` (`alignments_size` is memory's
                        `max_time`).
        """
        previous_alignments = state
        with tf.variable_scope(None, "Location_Sensitive_Attention", [query]):

            # processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
            processed_query = self.query_layer(query) if self.query_layer else query
            # -> [batch_size, 1, attention_dim]
            processed_query = tf.expand_dims(processed_query, 1)    #_bahdanau_score처럼 _location_sensitive_score에서 해도 되는데, 여기서 했네.

            # processed_location_features shape [batch_size, max_time, attention dimension]
            # [batch_size, max_time] -> [batch_size, max_time, 1]
            expanded_alignments = tf.expand_dims(previous_alignments, axis=2)    
            
            # location features [batch_size, max_time, filters]  # filters = 32로 고정
            f = self.location_convolution(expanded_alignments)
            
            # Projected location features [batch_size, max_time, attention_dim]
            processed_location_features = self.location_layer(f)

            # energy shape [batch_size, max_time]
            energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)


        # alignments shape = energy shape = [batch_size, max_time]
        alignments = self._probability_fn(energy, previous_alignments)   # location sensitivity attention에서는 softmax가 아니고, smoothing normalization function을 사용한다.

        # Cumulate alignments
        if self._cumulate:
            next_state = alignments + previous_alignments
        else:
            next_state = alignments

        return alignments, next_state


def _location_sensitive_score(W_query, W_fil, W_keys):
    """Impelements Bahdanau-style (cumulative) scoring function.
    This attention is described in:
            J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
      gio, �쏛ttention-based models for speech recognition,�� in Ad-
      vances in Neural Information Processing Systems, 2015, pp.
      577��585.

    #############################################################################
                      hybrid attention (content-based + location-based)
                                                       f = F * 慣_{i-1}
       energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
    #############################################################################

    Args:
            W_query: Tensor, shape '[batch_size, 1, attention_dim]' to compare to location features.
            W_location: processed previous alignments into location features, shape '[batch_size, max_time, attention_dim]'
            W_keys: Tensor, shape '[batch_size, max_time, attention_dim]', typically the encoder outputs.
    Returns:
            A '[batch_size, max_time]' attention score (energy)
    """
    # Get the number of hidden units from the trailing dimension of keys
    dtype = W_query.dtype
    num_units = W_keys.shape[-1].value or tf.shape(W_keys)[-1]

    v_a = tf.get_variable(
            'attention_variable', shape=[num_units], dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer())
    b_a = tf.get_variable(
            'attention_bias', shape=[num_units], dtype=dtype,
            initializer=tf.zeros_initializer())

    return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])


def _smoothing_normalization(e):
    """Applies a smoothing normalization function instead of softmax
    Introduced in:
            J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
      gio, �쏛ttention-based models for speech recognition,�� in Ad-
      vances in Neural Information Processing Systems, 2015, pp.
      577��585.

    ############################################################################
                                            Smoothing normalization function
                            a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
    ############################################################################

    Args:
            e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
                    values of an attention mechanism
    Returns:
            matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
                    attendance to multiple memory time steps.
    """
    return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)

class GmmAttention(AttentionMechanism):
    def __init__(self,
                 num_mixtures,
                 memory,
                 memory_sequence_length=None,
                 check_inner_dims_defined=True,
                 score_mask_value=None,
                 name='GmmAttention'):

        self.dtype = memory.dtype
        self.num_mixtures = num_mixtures
        self.query_layer = tf.layers.Dense(3 * num_mixtures, name='gmm_query_layer', use_bias=True, dtype=self.dtype)

        with tf.name_scope(name, 'GmmAttentionMechanismInit'):
            if score_mask_value is None:
                score_mask_value = 0.
            self._maybe_mask_score = functools.partial(
                _maybe_mask_score,
                memory_sequence_length=memory_sequence_length,
                score_mask_value=score_mask_value)
            self._value = _prepare_memory(
                memory, memory_sequence_length, check_inner_dims_defined)
            self._batch_size = (
                self._value.shape[0].value or tf.shape(self._value)[0])
            self._alignments_size = (
                    self._value.shape[1].value or tf.shape(self._value)[1])

    @property
    def values(self):
        return self._value

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def alignments_size(self):
        return self._alignments_size

    @property
    def state_size(self):
        return self.num_mixtures

    def initial_alignments(self, batch_size, dtype):
        max_time = self._alignments_size
        return _zero_state_tensors(max_time, batch_size, dtype)

    def initial_state(self, batch_size, dtype):
        state_size_ = self.state_size
        return _zero_state_tensors(state_size_, batch_size, dtype)

    def __call__(self, query, state):
        with tf.variable_scope("GmmAttention"):
            previous_kappa = state
            
            params = self.query_layer(query)   # query(dec_rnn_size=256) , params(num_mixtures(256)*3)
            alpha_hat, beta_hat, kappa_hat = tf.split(params, num_or_size_splits=3, axis=1)

            # [batch_size, num_mixtures, 1]
            alpha = tf.expand_dims(tf.exp(alpha_hat), axis=2)
            # softmax makes the alpha value more stable.
            # alpha = tf.expand_dims(tf.nn.softmax(alpha_hat, axis=1), axis=2)
            beta = tf.expand_dims(tf.exp(beta_hat), axis=2)
            kappa = tf.expand_dims(previous_kappa + tf.exp(kappa_hat), axis=2)

            # [1, 1, max_input_steps]
            mu = tf.reshape(tf.cast(tf.range(self.alignments_size), dtype=tf.float32), shape=[1, 1, self.alignments_size])  # [[[0,1,2,...]]]

            # [batch_size, max_input_steps]
            phi = tf.reduce_sum(alpha * tf.exp(-beta * (kappa - mu) ** 2.), axis=1)

        alignments = self._maybe_mask_score(phi)
        state = tf.squeeze(kappa, axis=2)

        return alignments, state
