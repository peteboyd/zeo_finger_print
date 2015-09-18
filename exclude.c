#define NPY_NO_DEPRECIATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>

int intersect_voronoi_nodes(double*, double*, double*, double*, double*, PyArrayObject*, PyArrayObject*);
static PyObject *compute_collision_array(PyObject*, PyObject*);

static PyMethodDef _methods[] = {
    {"compute_collision_array", (PyCFunction)compute_collision_array, METH_VARARGS, NULL},
    {NULL, NULL}
};


#if PY_MAJOR_VERSION >= 3
    #define MOD_ERROR_VAL NULL
    #define MOD_SUCCESS_VAL(val) val
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
        static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, };\
            ob = PyModule_Create(&moduledef);\
            import_array();
    #define PyInt_FromLong PyLong_FromLong
#else
    #define MOD_ERROR_VAL
    #define MOD_SUCCESS_VAL(val)
    #define MOD_INIT(name) void init##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
        ob = Py_InitModule3(name, methods, doc);\
        import_array();
#endif

MOD_INIT(SphereCollision)
{
    PyObject *m;
    MOD_DEF(m, "SphereCollision", "Computes line segment collisions with voronoi nodes",
            _methods)
    if (m==NULL)
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}


int intersect_voronoi_nodes(double* p1, double* p2, double* p3, double* p2p1, double* v1, PyArrayObject* vor_nodes, PyArrayObject* radii){

    npy_intp* vor_shape;
    PyObject* arr_item;
    double d, rad, u, div, len;
    void * ind;
    int j, k, array_val;
    vor_shape = PyArray_DIMS(vor_nodes);
    array_val = 0;
    for (j = 0; j< vor_shape[0]; j++){
        u = 0.0;
        div = 0.0;
        for (k=0; k<vor_shape[1]; k++){
            ind = PyArray_GETPTR2(vor_nodes, (npy_intp) j, (npy_intp) k);
            arr_item = PyArray_GETITEM(vor_nodes, (char*) ind);
            d = PyFloat_AsDouble(arr_item);
            v1[k] = d;
            u += (d - p1[k])*p2p1[k];
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
                //d = (u*p2p1[k] + p1[k] - v1[k]);
                d = (u*p2p1[k] + p1[k] - v1[k]);
                len += d*d;
            }
            len = sqrt(len);
            if (len <= rad){
               array_val = 1;
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
    PyObject* val;
    PyArrayObject* collision_array = NULL;

    if (!PyArg_ParseTuple(args, "OOO",
                          &atoms,
                          &vor_nodes,
                          &radii)){
        return NULL;
    };

    npy_intp dims[2]; 
    npy_intp* shape = PyArray_DIMS(atoms);
    dims[0] = shape[0];
    dims[1] = shape[0];
    int i, j, k, intersect;
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
    collision_array = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_INT, 0);

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
            intersect = intersect_voronoi_nodes(p1, p2, p3, p2p1, v1, vor_nodes, radii);
            val=PyInt_FromLong(intersect);
            ind1 = PyArray_GETPTR2(collision_array, (npy_intp) i, (npy_intp) j);
            PyArray_SETITEM(collision_array, (char*) ind1, val);
            Py_DECREF(val);
            val=PyInt_FromLong(intersect);
            ind1 = PyArray_GETPTR2(collision_array, (npy_intp) j, (npy_intp) i);
            PyArray_SETITEM(collision_array, (char*) ind1, val);
            Py_DECREF(val);
        }
    }
    free(p1);
    free(p2);
    free(p3);
    free(v1);
    free(p2p1);
    return PyArray_Return(collision_array);
}
