#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL QuantityDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"
#include "quantitydtype.h"
#include "casts.h"

static NPY_CASTING
resolve_descriptors(
    PyObject *NPY_UNUSED(self),
    PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
    PyArray_Descr *given_descrs[2],
    PyArray_Descr *loop_descrs[2],
    npy_intp *view_offset)
{
    if (given_descrs[1] == NULL)
    {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    }
    else
    {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    *view_offset = 0;
    return NPY_NO_CASTING;
}

typedef struct
{
    NpyAuxData base;
} conv_auxdata;

static int
cast_contiguous(
    PyArrayMethod_Context *NPY_UNUSED(context),
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const NPY_UNUSED(strides[]),
    conv_auxdata *NPY_UNUSED(auxdata))
{
    double *in = (double *)data[0];
    double *out = (double *)data[1];
    for (npy_intp N = dimensions[0]; N > 0; N--, in++, out++)
        *out = *in;
    return 0;
}

static int
cast_unaligned(
    PyArrayMethod_Context *NPY_UNUSED(context),
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    conv_auxdata *auxdata)
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];
    while (N--)
    {
        double in_val;
        double out_val;
        memcpy(&in_val, in, sizeof(double));
        out_val = in_val;
        memcpy(out, &out_val, sizeof(double));
        in += in_stride;
        out += out_stride;
    }
    return 0;
}

static int
get_loop(
    PyArrayMethod_Context *context,
    int aligned,
    int NPY_UNUSED(move_references),
    const npy_intp *strides,
    PyArrayMethod_StridedLoop **out_loop,
    NpyAuxData **out_transferdata,
    NPY_ARRAYMETHOD_FLAGS *flags)
{
    conv_auxdata *conv_auxdata;
    *out_transferdata = (NpyAuxData *)conv_auxdata;
    if (aligned)
        *out_loop = (PyArrayMethod_StridedLoop *)&cast_contiguous;
    else
        *out_loop = (PyArrayMethod_StridedLoop *)&cast_unaligned;
    *flags = 0;
    return 0;
}

static PyArray_DTypeMeta *dtypes[2] = {NULL, NULL};

static PyType_Slot slots[] = {
    {NPY_METH_resolve_descriptors, &resolve_descriptors},
    {_NPY_METH_get_loop, &get_loop},
    {0, NULL}};

PyArrayMethod_Spec CastSpec = {
    .name = "cast_QuantityDType",
    .nin = 1,
    .nout = 1,
    .flags = NPY_METH_SUPPORTS_UNALIGNED,
    .casting = NPY_SAFE_CASTING,
    .dtypes = dtypes,
    .slots = slots,
};
