#define NPY_NO_DEPRECIATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>
struct module_state{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif
int intersect_voronoi_nodes(double*, double*, double*, double*, double*, PyArrayObject*, PyArrayObject*);
static PyObject *compute_collision_array(PyObject*, PyObject*);


static PyObject*
error_out(PyObject *m){
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static PyMethodDef _methods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {"compute_collision_array", (PyCFunction)compute_collision_array, METH_VARARGS, NULL},
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

int intersect_voronoi_nodes(double* p1, double* p2, double* p3, double* p2p1, double* v1, PyArrayObject* vor_nodes, PyArrayObject* radii){

    npy_intp* vor_shape;
    PyObject* arr_item;
    double d, rad, u, div, len;
    void * ind;
    int j, k, array_val;
    vor_shape = PyArray_DIMS(vor_nodes);
    array_val = 1;
    for (j = 0; j< vor_shape[0]; j++){
        u = 0.0;
        div = 0.0;
        for (k=0; k<vor_shape[1]; k++){
            ind = PyArray_GETPTR2(vor_nodes, (npy_intp) j, (npy_intp) k);
            arr_item = PyArray_GETITEM(vor_nodes, (char*) ind);
            d = PyFloat_AsDouble(arr_item);
            v1[k] = d;
            u += (d - p1[k]);
            div += p2p1[k]*p2p1[k];
            Py_DECREF(arr_item);
        }
        u /= div;
        ind = PyArray_GETPTR1(radii, (npy_intp) j);
        arr_item = PyArray_GETITEM(radii, (char*) ind);
        rad = PyFloat_AsDouble(arr_item);
        Py_DECREF(arr_item);
        if((u >= 0)&&(u <= 1)){
            len=0.0;
            for (k=0; k<vor_shape[1]; k++){
                d = (u*p2p1[k] + p1[k] - v1[k]);
                len += d*d;
            }
            len = sqrt(len);
            if (len <= rad){
               array_val = 0;
            }
        }

    }
    return array_val;
}

static PyObject *compute_collision_array(PyObject *self, PyObject *args)
{
    PyArrayObject* atoms;
    PyArrayObject* vor_nodes;
    PyArrayObject* radii;
    PyObject* arr_item1;
    PyObject* arr_item2;
    if (!PyArg_ParseTuple(args, "OOO",
                          &atoms,
                          &vor_nodes,
                          &radii)){
        return NULL;
    };

    npy_intp* shape;
    shape = PyArray_DIMS(atoms);
    int i, j, k;
    void* ind1;
    void* ind2;
    double d1, d2;
    double *p1;
    double *p2;
    double *p3;
    double *p2p1;
    double *v1;
    p1 = (double*)malloc(sizeof(double)*(int)shape[1]);
    p2 = (double*)malloc(sizeof(double)*(int)shape[1]);
    p3 = (double*)malloc(sizeof(double)*(int)shape[1]);
    v1 = (double*)malloc(sizeof(double)*(int)shape[1]);
    p2p1 = (double*)malloc(sizeof(double)*(int)shape[1]);
    for (i=0; i<shape[0]; i++){
        for (j=i+1; j<shape[0]; j++){
            for (k=0; k<shape[1]; k++){

                ind1 = PyArray_GETPTR2(atoms, (npy_intp) i, (npy_intp) k);
                ind2 = PyArray_GETPTR2(atoms, (npy_intp) j, (npy_intp) k);
                arr_item1 = PyArray_GETITEM(atoms, (char*) ind1);
                arr_item2 = PyArray_GETITEM(atoms, (char*) ind2);
                d1 = PyFloat_AsDouble(arr_item1);
                d2 = PyFloat_AsDouble(arr_item2);
                p1[k] = d1; 
                p2[k] = d2;
                p2p1[k] = d2-d1;
                Py_DECREF(arr_item1);
                Py_DECREF(arr_item2);
            }
        }
    }
    free(p1);
    free(p2);
    free(p3);
    free(v1);
    free(p2p1);
}
