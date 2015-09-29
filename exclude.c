#define NPY_NO_DEPRECIATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>

int intersect_voronoi_nodes(double*, double*, double*, double*, double*, PyArrayObject*, PyArrayObject*);
static PyObject *compute_collision_array(PyObject*, PyObject*);
static PyObject *compute_minimg_distances(PyObject*, PyObject*);
double vector_length(double*);
double* invert(double*);
double *min_img_vect(double*, double*, double*, double*);

static PyMethodDef _methods[] = {
    {"compute_collision_array", (PyCFunction)compute_collision_array, METH_VARARGS, NULL},
    {"minimg_distances", (PyCFunction)compute_minimg_distances, METH_VARARGS, NULL},
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
            arr_item = PyArray_GETITEM(vor_nodes, ind);
            d = PyFloat_AsDouble(arr_item);
            v1[k] = d;
            u += (d - p1[k])*p2p1[k];
            div += p2p1[k]*p2p1[k];
            Py_DECREF(arr_item);
        }
        u /= div;
        ind = PyArray_GETPTR1(radii, (npy_intp) j);
        arr_item = PyArray_GETITEM(radii, ind);
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
    PyArrayObject* py_cell;
    PyObject* arr_item1;
    PyObject* arr_item2;
    PyObject* val;
    PyArrayObject* collision_array = NULL;

    if (!PyArg_ParseTuple(args, "OOOO",
                          &atoms,
                          &vor_nodes,
                          &radii,
                          &py_cell)){
        return NULL;
    };

    npy_intp dims[2]; 
    npy_intp* shape = PyArray_DIMS(atoms);
    dims[0] = shape[0];
    dims[1] = shape[0];
    int i, j, k, intersect, counter;
    void* ind1;
    void* ind2;
    double d1, d2;
    double *p1;
    double *p2;
    double *p3;
    double *p2p1;
    double *v1;
    double cell[9];
    double* icell;
    counter=0;
    for (i=0; i<3; i++){
        for (j=0; j<3; j++){
            ind1 = PyArray_GETPTR2(py_cell, (npy_intp) i, (npy_intp) j);
            arr_item1 = PyArray_GETITEM(py_cell, ind1);
            d1 = PyFloat_AsDouble(arr_item1);
            cell[counter] = d1;
            counter++;
            Py_DECREF(arr_item1);
        }
    }
    icell = invert(cell);
    p1 = (double*)malloc(sizeof(double)*(int)shape[1]);
    p2 = (double*)malloc(sizeof(double)*(int)shape[1]);
    p3 = (double*)malloc(sizeof(double)*(int)shape[1]);
    v1 = (double*)malloc(sizeof(double)*(int)shape[1]);
    collision_array = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_INT, 0);
    for (i=0; i<shape[0]; i++){
        for (j=i+1; j<shape[0]; j++){
            for (k=0; k<shape[1]; k++){
                
                ind1 = PyArray_GETPTR2(atoms, (npy_intp) i, (npy_intp) k);
                ind2 = PyArray_GETPTR2(atoms, (npy_intp) j, (npy_intp) k);
                arr_item1 = PyArray_GETITEM(atoms, ind1);
                arr_item2 = PyArray_GETITEM(atoms, ind2);
                d1 = PyFloat_AsDouble(arr_item1);
                d2 = PyFloat_AsDouble(arr_item2);
                p1[k] = d1; 
                p2[k] = d2;
                Py_DECREF(arr_item1);
                Py_DECREF(arr_item2);
               
            }
            //p2 - p1
            p2p1 = min_img_vect(p2, p1, cell, icell);
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
    return PyArray_Return(collision_array);
}

double *min_img_vect(double *vect1, double *vect2, double* cell, double *icell){
    double a, b, c;
    static double retvec[3];

    a = icell[0]*(vect1[0] - vect2[0]) + icell[3]*(vect1[1] - vect2[1]) + icell[6]*(vect1[2] - vect2[2]);
    b = icell[1]*(vect1[0] - vect2[0]) + icell[4]*(vect1[1] - vect2[1]) + icell[7]*(vect1[2] - vect2[2]);
    c = icell[2]*(vect1[0] - vect2[0]) + icell[5]*(vect1[1] - vect2[1]) + icell[8]*(vect1[2] - vect2[2]);

    a = a - rint(a);
    b = b - rint(b);
    c = c - rint(c);

    retvec[0] = cell[0]*a + cell[3]*b + cell[6]*c;
    retvec[1] = cell[1]*a + cell[4]*b + cell[7]*c;
    retvec[2] = cell[2]*a + cell[5]*b + cell[8]*c;
    return retvec;
}

double *invert(double* cell){
    double d,r;
    static double icell[9];

    icell[0] = cell[4]*cell[8] - cell[5]*cell[7];
    icell[1] = cell[2]*cell[7] - cell[1]*cell[8];
    icell[2] = cell[1]*cell[5] - cell[2]*cell[4];
    icell[3] = cell[5]*cell[6] - cell[3]*cell[8];
    icell[4] = cell[0]*cell[8] - cell[2]*cell[6];
    icell[5] = cell[2]*cell[3] - cell[0]*cell[5];
    icell[6] = cell[3]*cell[7] - cell[4]*cell[6];
    icell[7] = cell[1]*cell[6] - cell[0]*cell[7];
    icell[8] = cell[0]*cell[4] - cell[1]*cell[3];

    d = cell[0]*icell[0] + cell[3]*icell[1] + cell[6]*icell[2];
    r=0.0;
    if(abs(d)>0.0)r=1.0/d;
    icell[0] = r*icell[0];
    icell[1] = r*icell[1];
    icell[2] = r*icell[2];
    icell[3] = r*icell[3];
    icell[4] = r*icell[4];
    icell[5] = r*icell[5];
    icell[6] = r*icell[6];
    icell[7] = r*icell[7];
    icell[8] = r*icell[8];
    return icell;
}

double vector_length(double *vect){
    double vsq, val;
    vsq = vect[0]*vect[0] + vect[1]*vect[1] + vect[2]*vect[2];
    if(vsq > 0.0){
        val=sqrt(vsq);
    }
    else{
        val=0.0;
    }
    return val;
}

static PyObject *compute_minimg_distances(PyObject *self, PyObject *args)
{
    PyArrayObject* vectors1;
    PyArrayObject* vectors2;
    PyArrayObject* py_cell;
    PyObject* arr_item1;
    PyObject* arr_item2;
    PyObject* val;
    PyArrayObject* distmatrix;
    if (!PyArg_ParseTuple(args, "OOO",
                          &vectors1,
                          &vectors2,
                          &py_cell)){
        return NULL;
    };
    int i, j, k, counter;
    void* ind1;
    void* ind2;
    double dist, d1;
    double vect1[3], vect2[3];
    double cell[9];
    double* icell;
    double* min_vect;
    npy_intp dims[2]; 
    counter=0;
    for (i=0; i<3; i++){
        for (j=0; j<3; j++){
            ind1 = PyArray_GETPTR2(py_cell, (npy_intp) i, (npy_intp) j);
            arr_item1 = PyArray_GETITEM(py_cell, ind1);
            d1 = PyFloat_AsDouble(arr_item1);
            cell[counter] = d1;
            counter++;
            Py_DECREF(arr_item1);
        }
    }
    icell = invert(cell);

    npy_intp* shape1 = PyArray_DIMS(vectors1);
    npy_intp* shape2 = PyArray_DIMS(vectors2);

    dims[0] = shape1[0];
    dims[1] = shape2[0];
    distmatrix = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_FLOAT64, 0);
    for(i=0; i<(int)shape1[0]; i++){
        for (j=0; j<(int)shape2[0]; j++){
            //measure distances
            for(k=0; k<3; k++){
                ind1 = PyArray_GETPTR2(vectors1, (npy_intp) i, (npy_intp) k);
                ind2 = PyArray_GETPTR2(vectors2, (npy_intp) j, (npy_intp) k);
                arr_item1 = PyArray_GETITEM(vectors1, ind1);
                arr_item2 = PyArray_GETITEM(vectors2, ind2);
                vect1[k] = PyFloat_AsDouble(arr_item1);
                vect2[k] = PyFloat_AsDouble(arr_item2);
                Py_DECREF(arr_item1);
                Py_DECREF(arr_item2);
            }
            min_vect = min_img_vect(vect1, vect2, cell, icell);
            dist = vector_length(min_vect);
            val=PyFloat_FromDouble(dist);
            ind1 = PyArray_GETPTR2(distmatrix, (npy_intp) i, (npy_intp) j);
            PyArray_SETITEM(distmatrix, (char*) ind1, val);
            Py_DECREF(val);
        }
    }
    return PyArray_Return(distmatrix);
}
