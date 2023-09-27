import numpy as np

from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import LoadKeOps
from pykeops.common.operations import preprocess, postprocess
from pykeops.common.parse_type import get_sizes, complete_aliases, get_optional_flags
from pykeops.common.utils import axis2cat
from pykeops.numpy import default_dtype


class Genred:
    r"""
    Creates a new generic operation.

    This is KeOps' main function, whose usage is documented in
    the :doc:`user-guide <../../Genred>`,
    the :doc:`gallery of examples <../../../_auto_examples/index>`
    and the :doc:`high-level tutorials <../../../_auto_tutorials/index>`.
    Taking as input a handful of strings and integers that specify
    a custom Map-Reduce operation, it returns a C++ wrapper
    that can be called just like any other NumPy function.


    Note:
        On top of the **Sum** and **LogSumExp** reductions, KeOps
        supports
        :ref:`variants of the ArgKMin reduction <part.reduction>`
        that can be used
        to implement k-nearest neighbor search.
        These routines return indices encoded as **floating point numbers**, and
        produce no gradient. Fortunately though, you can simply
        turn them into ``LongTensors`` and use them to index
        your arrays, as showcased in the documentation
        of :func:`generic_argmin() <pykeops.numpy.generic_argmin>`, :func:`generic_argkmin() <pykeops.numpy.generic_argkmin>` and in the
        :doc:`K-means tutorial <../../../_auto_tutorials/kmeans/plot_kmeans_numpy>`.

    Example:
        >>> my_conv = Genred('Exp(-SqNorm2(x - y))',  # formula
        ...                  ['x = Vi(3)',            # 1st input: dim-3 vector per line
        ...                   'y = Vj(3)'],           # 2nd input: dim-3 vector per column
        ...                  reduction_op='Sum',      # we also support LogSumExp, Min, etc.
        ...                  axis=1)                  # reduce along the lines of the kernel matrix
        >>> # Apply it to 2d arrays x and y with 3 columns and a (huge) number of lines
        >>> x = np.random.randn(1000000, 3)
        >>> y = np.random.randn(2000000, 3)
        >>> a = my_conv(x, y)  # a_i = sum_j exp(-|x_i-y_j|^2)
        >>> print(a.shape)
        [1000000, 1]

    """

    def __init__(
        self,
        formula,
        aliases,
        reduction_op="Sum",
        axis=0,
        dtype=default_dtype,
        opt_arg=None,
        formula2=None,
        cuda_type=None,
        dtype_acc="auto",
        use_double_acc=False,
        sum_scheme="auto",
        enable_chunks=True,
        optional_flags=[],
        rec_multVar_highdim=None,
    ):
        r"""
        Instantiate a new generic operation.

        Note:
            :class:`Genred` relies on C++ or CUDA kernels that are compiled on-the-fly,
            and stored in a :ref:`cache directory <part.cache>` as shared libraries (".so" files) for later use.

        Args:
            formula (string): The scalar- or vector-valued expression
                that should be computed and reduced.
                The correct syntax is described in the :doc:`documentation <../../Genred>`,
                using appropriate :doc:`mathematical operations <../../../api/math-operations>`.
            aliases (list of strings): A list of identifiers of the form ``"AL = TYPE(DIM)"``
                that specify the categories and dimensions of the input variables. Here:

                  - ``AL`` is an alphanumerical alias, used in the **formula**.
                  - ``TYPE`` is a *category*. One of:

                    - ``Vi``: indexation by :math:`i` along axis 0.
                    - ``Vj``: indexation by :math:`j` along axis 1.
                    - ``Pm``: no indexation, the input tensor is a *vector* and not a 2d array.

                  - ``DIM`` is an integer, the dimension of the current variable.

                As described below, :meth:`__call__` will expect as input Tensors whose
                shape are compatible with **aliases**.

        Keyword Args:
            reduction_op (string, default = ``"Sum"``): Specifies the reduction
                operation that is applied to reduce the values
                of ``formula(x_i, y_j, ...)`` along axis 0 or axis 1.
                The supported values are one of :ref:`part.reduction`

            axis (int, default = 0): Specifies the dimension of the "kernel matrix" that is reduced by our routine.
                The supported values are:

                  - **axis** = 0: reduction with respect to :math:`i`, outputs a ``Vj`` or ":math:`j`" variable.
                  - **axis** = 1: reduction with respect to :math:`j`, outputs a ``Vi`` or ":math:`i`" variable.

            dtype (string, default = ``"float64"``): Specifies the numerical ``dtype`` of the input and output arrays.
                The supported values are:

                  - **dtype** = ``"float32"``.
                  - **dtype** = ``"float64"``.

            opt_arg (int, default = None): If **reduction_op** is in ``["KMin", "ArgKMin", "KMinArgKMin"]``,
                this argument allows you to specify the number ``K`` of neighbors to consider.

            dtype_acc (string, default ``"auto"``): type for accumulator of reduction, before casting to dtype.
                It improves the accuracy of results in case of large sized data, but is slower.
                Default value "auto" will set this option to the value of dtype. The supported values are:

                  - **dtype_acc** = ``"float16"`` : allowed only if dtype is "float16".
                  - **dtype_acc** = ``"float32"`` : allowed only if dtype is "float16" or "float32".
                  - **dtype_acc** = ``"float64"`` : allowed only if dtype is "float32" or "float64"..

            use_double_acc (bool, default False): same as setting dtype_acc="float64" (only one of the two options can be set)
                If True, accumulate results of reduction in float64 variables, before casting to float32.
                This can only be set to True when data is in float32 or float64.
                It improves the accuracy of results in case of large sized data, but is slower.

            sum_scheme (string, default ``"auto"``): method used to sum up results for reductions. This option may be changed only
                when reduction_op is one of: "Sum", "MaxSumShiftExp", "LogSumExp", "Max_SumShiftExpWeight", "LogSumExpWeight", "SumSoftMaxWeight".
                Default value "auto" will set this option to "block_red" for these reductions. Possible values are:
                  - **sum_scheme** =  ``"direct_sum"``: direct summation
                  - **sum_scheme** =  ``"block_sum"``: use an intermediate accumulator in each block before accumulating in the output. This improves accuracy for large sized data.
                  - **sum_scheme** =  ``"kahan_scheme"``: use Kahan summation algorithm to compensate for round-off errors. This improves
                accuracy for large sized data.

            enable_chunks (bool, default True): enable automatic selection of special "chunked" computation mode for accelerating reductions
                                with formulas involving large dimension variables.

                        optional_flags (list, default []): further optional flags passed to the compiler, in the form ['-D...=...','-D...=...']

        """
        if cuda_type:
            # cuda_type is just old keyword for dtype, so this is just a trick to keep backward compatibility
            dtype = cuda_type

        if dtype in ("float16", "half"):
            raise ValueError(
                "[KeOps] Float16 type is only supported with PyTorch tensors inputs."
            )

        self.reduction_op = reduction_op
        reduction_op_internal, formula2 = preprocess(reduction_op, formula2)

        if rec_multVar_highdim is not None:
            optional_flags += ["-DMULT_VAR_HIGHDIM=1"]

        self.optional_flags = optional_flags + get_optional_flags(
            reduction_op_internal,
            dtype_acc,
            use_double_acc,
            sum_scheme,
            dtype,
            enable_chunks,
        )
        str_opt_arg = "," + str(opt_arg) if opt_arg else ""
        str_formula2 = "," + formula2 if formula2 else ""

        self.formula = (
            reduction_op_internal
            + "_Reduction("
            + formula
            + str_opt_arg
            + ","
            + str(axis2cat(axis))
            + str_formula2
            + ")"
        )
        self.aliases = complete_aliases(self.formula, aliases)
        self.dtype = dtype
        self.myconv = LoadKeOps(
            self.formula, self.aliases, self.dtype, "numpy", self.optional_flags
        ).import_module()
        self.axis = axis
        self.opt_arg = opt_arg

    def __call__(self, *args, backend="auto", device_id=-1, ranges=None):
        r"""
        Apply the routine on arbitrary NumPy arrays.

        .. warning::
            Even for variables of size 1 (e.g. :math:`a_i\in\mathbb{R}`
            for :math:`i\in[0,M)`), KeOps expects inputs to be formatted
            as 2d Tensors of size ``(M,dim)``. In practice,
            ``a.view(-1,1)`` should be used to turn a vector of weights
            into a *list of scalar values*.


        Args:
            *args (2d arrays (variables ``Vi(..)``, ``Vj(..)``) and 1d arrays (parameters ``Pm(..)``)): The input numerical arrays,
                which should all have the same ``dtype``, be **contiguous** and be stored on
                the **same device**. KeOps expects one array per alias,
                with the following compatibility rules:

                    - All ``Vi(Dim_k)`` variables are encoded as **2d-arrays** with ``Dim_k`` columns and the same number of lines :math:`M`.
                    - All ``Vj(Dim_k)`` variables are encoded as **2d-arrays** with ``Dim_k`` columns and the same number of lines :math:`N`.
                    - All ``Pm(Dim_k)`` variables are encoded as **1d-arrays** (vectors) of size ``Dim_k``.

        Keyword Args:
            backend (string): Specifies the map-reduce scheme.
                The supported values are:

                    - ``"auto"`` (default): let KeOps decide which backend is best suited to your data, based on the tensors' shapes. ``"GPU_1D"`` will be chosen in most cases.
                    - ``"CPU"``: use a simple C++ ``for`` loop on a single CPU core.
                    - ``"GPU_1D"``: use a `simple multithreading scheme <https://github.com/getkeops/keops/blob/master/keops/core/GpuConv1D.cu>`_ on the GPU - basically, one thread per value of the output index.
                    - ``"GPU_2D"``: use a more sophisticated `2D parallelization scheme <https://github.com/getkeops/keops/blob/master/keops/core/GpuConv2D.cu>`_ on the GPU.
                    - ``"GPU"``: let KeOps decide which one of the ``"GPU_1D"`` or the ``"GPU_2D"`` scheme will run faster on the given input.

            device_id (int, default=-1): Specifies the GPU that should be used
                to perform the computation; a negative value lets your system
                choose the default GPU. This parameter is only useful if your
                system has access to several GPUs.

            ranges (6-uple of integer arrays, None by default):
                Ranges of integers that specify a
                :doc:`block-sparse reduction scheme <../../sparsity>`
                with *Mc clusters along axis 0* and *Nc clusters along axis 1*.
                If None (default), we simply loop over all indices
                :math:`i\in[0,M)` and :math:`j\in[0,N)`.

                **The first three ranges** will be used if **axis** = 1
                (reduction along the axis of ":math:`j` variables"),
                and to compute gradients with respect to ``Vi(..)`` variables:

                    - ``ranges_i``, (Mc,2) integer array - slice indices
                      :math:`[\operatorname{start}^I_k,\operatorname{end}^I_k)` in :math:`[0,M]`
                      that specify our Mc blocks along the axis 0
                      of ":math:`i` variables".
                    - ``slices_i``, (Mc,) integer array - consecutive slice indices
                      :math:`[\operatorname{end}^S_1, ..., \operatorname{end}^S_{M_c}]`
                      that specify Mc ranges :math:`[\operatorname{start}^S_k,\operatorname{end}^S_k)` in ``redranges_j``,
                      with :math:`\operatorname{start}^S_k = \operatorname{end}^S_{k-1}`.
                      **The first 0 is implicit**, meaning that :math:`\operatorname{start}^S_0 = 0`, and we typically expect that
                      ``slices_i[-1] == len(redrange_j)``.
                    - ``redranges_j``, (Mcc,2) integer array - slice indices
                      :math:`[\operatorname{start}^J_\ell,\operatorname{end}^J_\ell)` in :math:`[0,N]`
                      that specify reduction ranges along the axis 1
                      of ":math:`j` variables".

                If **axis** = 1, these integer arrays allow us to say that ``for k in range(Mc)``, the output values for
                indices ``i in range( ranges_i[k,0], ranges_i[k,1] )`` should be computed using a Map-Reduce scheme over
                indices ``j in Union( range( redranges_j[l, 0], redranges_j[l, 1] ))`` for ``l in range( slices_i[k-1], slices_i[k] )``.

                **Likewise, the last three ranges** will be used if **axis** = 0
                (reduction along the axis of ":math:`i` variables"),
                and to compute gradients with respect to ``Vj(..)`` variables:

                    - ``ranges_j``, (Nc,2) integer array - slice indices
                      :math:`[\operatorname{start}^J_k,\operatorname{end}^J_k)` in :math:`[0,N]`
                      that specify our Nc blocks along the axis 1
                      of ":math:`j` variables".
                    - ``slices_j``, (Nc,) integer array - consecutive slice indices
                      :math:`[\operatorname{end}^S_1, ..., \operatorname{end}^S_{N_c}]`
                      that specify Nc ranges :math:`[\operatorname{start}^S_k,\operatorname{end}^S_k)` in ``redranges_i``,
                      with :math:`\operatorname{start}^S_k = \operatorname{end}^S_{k-1}`.
                      **The first 0 is implicit**, meaning that :math:`\operatorname{start}^S_0 = 0`, and we typically expect that
                      ``slices_j[-1] == len(redrange_i)``.
                    - ``redranges_i``, (Ncc,2) integer array - slice indices
                      :math:`[\operatorname{start}^I_\ell,\operatorname{end}^I_\ell)` in :math:`[0,M]`
                      that specify reduction ranges along the axis 0
                      of ":math:`i` variables".

                If **axis** = 0,
                these integer arrays allow us to say that ``for k in range(Nc)``, the output values for
                indices ``j in range( ranges_j[k,0], ranges_j[k,1] )`` should be computed using a Map-Reduce scheme over
                indices ``i in Union( range( redranges_i[l, 0], redranges_i[l, 1] ))`` for ``l in range( slices_j[k-1], slices_j[k] )``.

        Returns:
            (M,D) or (N,D) array:

            The output of the reduction,
            a **2d-tensor** with :math:`M` or :math:`N` lines (if **axis** = 1
            or **axis** = 0, respectively) and a number of columns
            that is inferred from the **formula**.
        """

        # Get tags
        tagCpuGpu, tag1D2D, _ = get_tag_backend(backend, args)
        if ranges is None:
            ranges = ()  # To keep the same type

        # N.B.: KeOps C++ expects contiguous integer arrays as ranges
        ranges = tuple(np.ascontiguousarray(r) for r in ranges)

        nx, ny = get_sizes(self.aliases, *args)
        nout, nred = (nx, ny) if self.axis == 1 else (ny, nx)

        if "Arg" in self.reduction_op:
            # when using Arg type reductions,
            # if nred is greater than 16 millions and dtype=float32, the result is not reliable
            # because we encode indices as floats, so we raise an exception ;
            # same with float16 type and nred>2048
            if nred > 1.6e7 and self.dtype in ("float32", "float"):
                raise ValueError(
                    "size of input array is too large for Arg type reduction with single precision. Use double precision."
                )
            elif nred > 2048 and self.dtype in ("float16", "half"):
                raise ValueError(
                    "size of input array is too large for Arg type reduction with float16 dtype.."
                )

        out = self.myconv.genred_numpy(
            tagCpuGpu, tag1D2D, 0, device_id, ranges, nx, ny, *args
        )

        return postprocess(
            out, "numpy", self.reduction_op, nout, self.opt_arg, self.dtype
        )
