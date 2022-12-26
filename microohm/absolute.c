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

d_absolute_strided_loop(
    PyArrayMethod_Context *context,
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *out = data[1];
    npy_intp in1_stride = strides[0];
    npy_intp out_stride = strides[1];
    while (N--)
    {
        *(double *)out = fabs(*(double *)in1);
        in1 += in1_stride;
        out += out_stride;
    }
    return 0;
}

q_absolute_resolve_descriptors(
    PyObject *self,
    PyArray_DTypeMeta *dtypes[],
    PyArray_Descr *given_descrs[],
    PyArray_Descr *loop_descrs[],
    npy_intp *unused)
{
    PyObject *units1 = ((QuantityDTypeObject *)given_descrs[0])->units;
    loop_descrs[1] = instantiate_QuantityDType(units1);
    if (loop_descrs[1] == NULL)
        return -1;
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    return NPY_NO_CASTING;
}

int init_absolute_ufunc(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL)
        return -1;
    PyObject *absolute = PyObject_GetAttrString(numpy, "absolute");
    Py_DECREF(numpy);
    if (absolute == NULL)
        return -1;
    static PyArray_DTypeMeta *dtypes[2] = {&QuantityDType, &QuantityDType};
    static PyType_Slot slots[] = {
        {NPY_METH_resolve_descriptors, &q_absolute_resolve_descriptors},
        {NPY_METH_strided_loop, &d_absolute_strided_loop},
        {0, NULL}};
    PyArrayMethod_Spec q_AbsoluteSpec = {
        .name = "q_absolute",
        .nin = 1,
        .nout = 1,
        .dtypes = dtypes,
        .slots = slots,
        .flags = 0,
        .casting = NPY_NO_CASTING,
    };
    if (PyUFunc_AddLoopFromSpec(absolute, &q_AbsoluteSpec) < 0)
    {
        Py_DECREF(absolute);
        return -1;
    }
    Py_DECREF(absolute);
    return 0;
}
