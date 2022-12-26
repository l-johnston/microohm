#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL QuantityDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/experimental_dtype_api.h"
#include "quantitydtype.h"
#include "umath.h"

static int
dd_multiply_strided_loop(
    PyArrayMethod_Context *context,
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];
    while (N--)
    {
        *(double *)out = *(double *)in1 * *(double *)in2;
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }
    return 0;
}

static int
id_multiply_strided_loop(
    PyArrayMethod_Context *context,
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];
    while (N--)
    {
        *(double *)out = (double)*(int64_t *)in1 * *(double *)in2;
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }
    return 0;
}

static int
di_multiply_strided_loop(
    PyArrayMethod_Context *context,
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];
    while (N--)
    {
        *(double *)out = *(double *)in1 * (double)*(int64_t *)in2;
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }
    return 0;
}

static NPY_CASTING
qq_multiply_resolve_descriptors(
    PyObject *self,
    PyArray_DTypeMeta *dtypes[],
    PyArray_Descr *given_descrs[],
    PyArray_Descr *loop_descrs[],
    npy_intp *unused)
{
    PyObject *units1 = ((QuantityDTypeObject *)given_descrs[0])->units;
    PyObject *units2 = ((QuantityDTypeObject *)given_descrs[1])->units;
    PyObject *new_units = PyUnicode_Concat(PyUnicode_Concat(units1, PyUnicode_FromString("*")), units2);
    if (new_units == NULL)
        return -1;
    loop_descrs[2] = instantiate_QuantityDType(new_units);
    if (loop_descrs[2] == NULL)
        return -1;
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    return NPY_NO_CASTING;
}

nq_multiply_resolve_descriptors(
    PyObject *self,
    PyArray_DTypeMeta *dtypes[],
    PyArray_Descr *given_descrs[],
    PyArray_Descr *loop_descrs[],
    npy_intp *unused)
{
    loop_descrs[2] = instantiate_QuantityDType(((QuantityDTypeObject *)given_descrs[1])->units);
    if (loop_descrs[2] == NULL)
        return -1;
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    return NPY_SAFE_CASTING;
}

qn_multiply_resolve_descriptors(
    PyObject *self,
    PyArray_DTypeMeta *dtypes[],
    PyArray_Descr *given_descrs[],
    PyArray_Descr *loop_descrs[],
    npy_intp *unused)
{
    loop_descrs[2] = instantiate_QuantityDType(((QuantityDTypeObject *)given_descrs[0])->units);
    if (loop_descrs[2] == NULL)
        return -1;
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    return NPY_SAFE_CASTING;
}

int init_multiply_ufunc(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL)
        return -1;
    PyObject *multiply = PyObject_GetAttrString(numpy, "multiply");
    Py_DECREF(numpy);
    if (multiply == NULL)
        return -1;
    static PyArray_DTypeMeta *dtypes[3] = {&QuantityDType, &QuantityDType, &QuantityDType};
    static PyType_Slot slots[] = {
        {NPY_METH_resolve_descriptors, &qq_multiply_resolve_descriptors},
        {NPY_METH_strided_loop, &dd_multiply_strided_loop},
        {0, NULL}};
    PyArrayMethod_Spec qq_MultiplySpec = {
        .name = "qq_multiply",
        .nin = 2,
        .nout = 1,
        .dtypes = dtypes,
        .slots = slots,
        .flags = 0,
        .casting = NPY_NO_CASTING,
    };
    if (PyUFunc_AddLoopFromSpec(multiply, &qq_MultiplySpec) < 0)
    {
        Py_DECREF(multiply);
        return -1;
    }
    dtypes[0] = &PyArray_DoubleDType;
    slots[0].pfunc = &nq_multiply_resolve_descriptors;
    PyArrayMethod_Spec dq_MultiplySpec = {
        .name = "dq_multiply",
        .nin = 2,
        .nout = 1,
        .dtypes = dtypes,
        .slots = slots,
        .flags = 0,
        .casting = NPY_NO_CASTING,
    };
    if (PyUFunc_AddLoopFromSpec(multiply, &dq_MultiplySpec) < 0)
    {
        Py_DECREF(multiply);
        return -1;
    }
    dtypes[0] = &PyArray_Int64DType;
    slots[1].pfunc = &id_multiply_strided_loop;
    PyArrayMethod_Spec iq_MultiplySpec = {
        .name = "iq_multiply",
        .nin = 2,
        .nout = 1,
        .dtypes = dtypes,
        .slots = slots,
        .flags = 0,
        .casting = NPY_NO_CASTING,
    };
    if (PyUFunc_AddLoopFromSpec(multiply, &iq_MultiplySpec) < 0)
    {
        Py_DECREF(multiply);
        return -1;
    }
    dtypes[0] = &QuantityDType;
    dtypes[1] = &PyArray_DoubleDType;
    slots[0].pfunc = &qn_multiply_resolve_descriptors;
    slots[1].pfunc = dd_multiply_strided_loop;
    PyArrayMethod_Spec qd_MultiplySpec = {
        .name = "qd_multiply",
        .nin = 2,
        .nout = 1,
        .dtypes = dtypes,
        .slots = slots,
        .flags = 0,
        .casting = NPY_NO_CASTING,
    };
    if (PyUFunc_AddLoopFromSpec(multiply, &qd_MultiplySpec) < 0)
    {
        Py_DECREF(multiply);
        return -1;
    }
    dtypes[1] = &PyArray_Int64DType;
    slots[1].pfunc = di_multiply_strided_loop;
    PyArrayMethod_Spec qi_MultiplySpec = {
        .name = "qi_multiply",
        .nin = 2,
        .nout = 1,
        .dtypes = dtypes,
        .slots = slots,
        .flags = 0,
        .casting = NPY_NO_CASTING,
    };
    if (PyUFunc_AddLoopFromSpec(multiply, &qi_MultiplySpec) < 0)
    {
        Py_DECREF(multiply);
        return -1;
    }
    Py_DECREF(multiply);
    return 0;
}
