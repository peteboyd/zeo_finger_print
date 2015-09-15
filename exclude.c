#define NPY_NO_DEPRECIATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
struct module_state{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject*
error_out(PyObject *m){
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static PyMethodDef _methods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {NULL, NULL}
};


#if PY_MAJOR_VERSION >= 3
static int _traverse(PyObject *m, visitproc visit, void *arg){
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int _clear(PyObject *m){
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "sphere_exclude",
    NULL,
    sizeof(struct module_state),
    _methods,
    NULL,
    _traverse,
    _clear,
    NULL
};

#define INITERROR return NULL

PyObject*
PyInit_sphere_exclude(void)

#else
#define INITERROR return

void
initsphere_exclude(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("sphere_exclude", _methods);
#endif
    if (module==NULL)
        INITERROR;
    struct module_state *st=GETSTATE(module);

    st->error = PyErr_NewException("myextension.Error", NULL, NULL);
    if (st->error=NULL){
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}


