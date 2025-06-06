#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h" // For PyArrayObject, PyArray_DESCR
#include "numpy/npy_common.h"   // For npy_swap_bytes

#include <stdio.h> // For perror, fprintf, stderr if not fully removed
#include <stdlib.h> // For malloc, free, realloc, calloc, rand, srand
#include <math.h>   // For log2, pow, round, roundf
#include <string.h> // For memcpy
// #include <time.h> // For srand and time - consider replacing rand for NumPy

// Forward declaration for the main sorting routine
static int TBSort_generic(char *arr, npy_intp num, PyArray_CompareFunc *compare, int elsize, PyArrayObject *arr_obj);


/* Helper: Swap two elements of size elsize */
static void swap_elements(char *a, char *b, int elsize) {
    char *temp = (char *)PyMem_RawMalloc(elsize);
    if (!temp) {
        // No easy way to propagate this error back to npy_tbsort's return
        // This is a critical low-level utility; failure here is severe.
        // In a real NumPy integration, this might need a different error strategy
        // or rely on the fact that elsize is usually small.
        return;
    }
    memcpy(temp, a, elsize);
    memcpy(a, b, elsize);
    memcpy(b, temp, elsize);
    PyMem_RawFree(temp);
}

/* Helper: Clamp an integer n to be within lower and upper bounds */
static int myclamp(int n, int lower, int upper) {
    if (n < lower) return lower;
    if (n > upper) return upper;
    return n;
}

/*
 * Helper: Binary search for element e_ptr in array a_ptr of size n.
 * Returns the position of the element in the array that is <= e_ptr.
 * Assumes the array is sorted according to 'compare'.
 */
static npy_intp search_generic(char *a_ptr, npy_intp n, char *e_ptr, PyArray_CompareFunc *compare, int elsize, PyArrayObject *arr_obj) {
    npy_intp low = 0, high = n - 1;
    npy_intp ans = -1;

    while (low <= high) {
        npy_intp mid = low + (high - low) / 2;
        if (compare(a_ptr + mid * elsize, e_ptr, arr_obj) <= 0) {
            ans = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return ans;
}

/* Helper: Insertion sort for generic types */
static void insertionSort_generic(char *arr_ptr, npy_intp n, PyArray_CompareFunc *compare, int elsize, PyArrayObject *arr_obj) {
    if (n <= 1) return;
    npy_intp i, j;
    char *key = (char *)PyMem_RawMalloc(elsize);
    if (!key) {
        // Error handling: ideally propagate, but difficult from here
        return;
    }

    for (i = 1; i < n; i++) {
        memcpy(key, arr_ptr + i * elsize, elsize);
        j = i - 1;

        while (j >= 0 && compare(arr_ptr + j * elsize, key, arr_obj) > 0) {
            memcpy(arr_ptr + (j + 1) * elsize, arr_ptr + j * elsize, elsize);
            j = j - 1;
        }
        memcpy(arr_ptr + (j + 1) * elsize, key, elsize);
    }
    PyMem_RawFree(key);
}

// Structure for bins (elements will be char* for generic type)
typedef struct {
    char* elements;
    npy_intp size;
    npy_intp capacity;
} Bin_generic;


// Main TBSort logic adapted for generic types
// arr: pointer to the data to be sorted
// num: number of elements
// compare: comparison function
// elsize: size of each element in bytes
// arr_obj: PyArrayObject, passed for the comparison function
static int TBSort_generic(char *arr, npy_intp num, PyArray_CompareFunc *compare, int elsize, PyArrayObject *arr_obj) {
    if (num <= 1) {
        return 0; // Already sorted
    }

    // Handle very small arrays with insertion sort
    if (num < 4) {
        insertionSort_generic(arr, num, compare, elsize, arr_obj);
        return 0;
    }

    // 2. TREE Step
    npy_intp treeSize;
    if (num < 4) {
        treeSize = 1;
    } else {
        double log2_num = log2(num);
        if (log2_num <= 0) log2_num = 1;
        double log2_log2_num = log2(log2_num);
        if (log2_log2_num < 0) log2_log2_num = 0;
        treeSize = (npy_intp)pow(2, round(log2_log2_num));
    }

    if (treeSize < 2 && num > 1) {
        treeSize = 2;
    }
    if (num <= 1) {
        treeSize = 1;
    }
    if (treeSize > num) {
        treeSize = num;
    }

    char *sampleTree = (char *)PyMem_RawMalloc(treeSize * elsize);
    if (!sampleTree) {
        PyErr_NoMemory();
        return -1;
    }

    // Populate sampleTree with random elements from arr
    // TODO: Consider replacing rand() with a NumPy-approved RNG if necessary
    // For now, ensure srand is called once, perhaps in npy_tbsort or module init
    for (npy_intp i = 0; i < treeSize; i++) {
        if (num == 0) continue;
        memcpy(sampleTree + i * elsize, arr + (rand() % num) * elsize, elsize);
    }
    insertionSort_generic(sampleTree, treeSize, compare, elsize, arr_obj);

    // 3. BIN Step
    npy_intp binCount;
    double logVal = log2(num);
    if (logVal <= 0 || num < 2) {
        binCount = treeSize + 2;
    } else {
        binCount = (npy_intp)(num / logVal);
    }

    if (binCount < treeSize + 2) {
        binCount = treeSize + 2;
    }
    if (binCount == 0) binCount = 1;


    Bin_generic *bins = (Bin_generic *)PyMem_RawCalloc(binCount, sizeof(Bin_generic));
    if (!bins) {
        PyMem_RawFree(sampleTree);
        PyErr_NoMemory();
        return -1;
    }

    for (npy_intp i = 0; i < binCount; i++) {
        bins[i].capacity = 4; // Initial capacity for #elements, not bytes
        bins[i].elements = (char *)PyMem_RawMalloc(bins[i].capacity * elsize);
        if (!bins[i].elements) {
            for (npy_intp j = 0; j < i; j++) PyMem_RawFree(bins[j].elements);
            PyMem_RawFree(bins);
            PyMem_RawFree(sampleTree);
            PyErr_NoMemory();
            return -1;
        }
        bins[i].size = 0;
    }

    npy_intp targetbinSize = treeSize + 2;
    npy_intp *targetbin_indices = (npy_intp *)PyMem_RawMalloc(targetbinSize * sizeof(npy_intp));
    float *slope = (float *)PyMem_RawMalloc((targetbinSize - 1) * sizeof(float));
    float *offset = (float *)PyMem_RawMalloc((targetbinSize - 1) * sizeof(float));

    if (!targetbin_indices || !slope || !offset) {
         for (npy_intp i = 0; i < binCount; i++) PyMem_RawFree(bins[i].elements);
         PyMem_RawFree(bins);
         PyMem_RawFree(sampleTree);
         if(targetbin_indices) PyMem_RawFree(targetbin_indices);
         if(slope) PyMem_RawFree(slope);
         if(offset) PyMem_RawFree(offset);
         PyErr_NoMemory();
         return -1;
    }

    targetbin_indices[0] = 0;
    targetbin_indices[targetbinSize - 1] = binCount - 1;
    for (npy_intp i = 1; i < targetbinSize - 1; i++) {
        targetbin_indices[i] = myclamp((npy_intp)roundf(num * (i) / (float)(treeSize + 1) / logVal), 0, binCount - 1);
    }

    for (npy_intp i = 0; i < targetbinSize - 1; i++) {
        // This part is tricky without knowing the data type to cast to float.
        // The original TBSort assumes int. For generic sort, this linear model
        // for bin prediction based on value needs re-evaluation.
        // A simple approach for now: use element index as a proxy for value if elements are roughly uniform.
        // This is a placeholder and likely needs a more robust strategy for general types.
        // Or, the TBSort algorithm might be fundamentally suited for numerical types where such a mapping is feasible.
        // For now, let's assume the comparison function gives enough info for search, and this part is simplified.
        // This part of the original algorithm might not directly translate to generic comparison-based sort.
        // We will use a simplified binning strategy: divide elements based on rank against sampleTree.
        // For now, we will keep the structure but acknowledge this is a weak point for generic types.
        // The values x1, x2 from sampleTree are needed. This requires interpreting element values as float.
        // This is a major assumption. If the type is not numeric, this will fail.
        // For now, we'll proceed with a simplified logic that doesn't rely on float conversion of values.
        // The original TBSort's bin prediction (slope/offset) is highly type-dependent (assumes numeric).
        // We will simplify bin assignment to be based purely on `search_generic` results.
        // This means `slope` and `offset` might not be used as in the original paper if values aren't floats.
        // We'll use `mypos` from `search_generic` to determine bins more directly.
        // This part of the original code is being commented out as it's not type-generic.
        // float x1 = ...; float x2 = ...;
        // slope[i] = ...; offset[i] = ...;
        // A simpler binning:
        if (i < treeSize) { // treeSize segments based on sampleTree
             slope[i] = (float)binCount / (float)treeSize; // Distribute bins somewhat evenly across tree segments
             offset[i] = (float)i * binCount / (float)treeSize;
        } else { // Fallback for the last segment
             slope[i] = 0;
             offset[i] = (float)binCount -1; // Put in the last bin
        }
    }


    for (npy_intp i = 0; i < num; i++) {
        char *current_element_ptr = arr + i * elsize;
        npy_intp mypos = search_generic(sampleTree, treeSize, current_element_ptr, compare, elsize, arr_obj);

        npy_intp mybin_idx;
        // Simplified binning: assign to a bin based on which segment of sampleTree it falls into.
        if (mypos == -1) { // Smaller than smallest in sampleTree
            mybin_idx = 0;
        } else if (mypos == treeSize - 1 && compare(current_element_ptr, sampleTree + mypos * elsize, arr_obj) >= 0) { // Larger or equal to largest in sampleTree
            mybin_idx = binCount - 1;
        } else {
            // Distribute among other bins based on mypos. This is a simplification.
            // The original slope/offset was more nuanced.
            mybin_idx = myclamp((npy_intp)((mypos + 1.0) / treeSize * (binCount-2)) +1 , 0, binCount - 1) ;
        }


        if (bins[mybin_idx].size >= bins[mybin_idx].capacity) {
            bins[mybin_idx].capacity = (bins[mybin_idx].capacity == 0) ? 1 : bins[mybin_idx].capacity * 2;
            char* new_elements = (char*)PyMem_RawRealloc(bins[mybin_idx].elements, bins[mybin_idx].capacity * elsize);
            if (!new_elements) {
                // Cleanup and error
                PyMem_RawFree(sampleTree);
                for(npy_intp k=0; k<binCount; ++k) if(bins[k].elements) PyMem_RawFree(bins[k].elements);
                PyMem_RawFree(bins);
                PyMem_RawFree(targetbin_indices);
                PyMem_RawFree(slope);
                PyMem_RawFree(offset);
                PyErr_NoMemory();
                return -1;
            }
            bins[mybin_idx].elements = new_elements;
        }
        memcpy(bins[mybin_idx].elements + bins[mybin_idx].size * elsize, current_element_ptr, elsize);
        bins[mybin_idx].size++;
    }

    // 4. SORT Step
    npy_intp binThreshold = (npy_intp)(5 * num / (float)binCount); // Using num from TBSort_generic
    if (binCount == 0) binThreshold = num + 1;

    npy_intp curpos_byte = 0; // Current position in bytes in the original array
    for (npy_intp i = 0; i < binCount; i++) {
        if (bins[i].size == 0) {
            PyMem_RawFree(bins[i].elements);
            continue;
        }

        if (bins[i].size < binThreshold) {
            insertionSort_generic(bins[i].elements, bins[i].size, compare, elsize, arr_obj);
        } else {
            if (TBSort_generic(bins[i].elements, bins[i].size, compare, elsize, arr_obj) < 0) {
                // Error occurred in recursive call, propagate
                PyMem_RawFree(sampleTree);
                for(npy_intp k=0; k<binCount; ++k) if(bins[k].elements && (k>i || bins[k].size > 0)) PyMem_RawFree(bins[k].elements);
                PyMem_RawFree(bins);
                PyMem_RawFree(targetbin_indices);
                PyMem_RawFree(slope);
                PyMem_RawFree(offset);
                return -1; // Error already set by nested call
            }
        }

        if (curpos_byte + bins[i].size * elsize > num * elsize) {
             // This should not happen with correct logic
            PyErr_SetString(PyExc_RuntimeError, "TBSort: writing out of bounds.");
            PyMem_RawFree(sampleTree);
            for(npy_intp k=0; k<binCount; ++k) if(bins[k].elements) PyMem_RawFree(bins[k].elements);
            PyMem_RawFree(bins);
            PyMem_RawFree(targetbin_indices);
            PyMem_RawFree(slope);
            PyMem_RawFree(offset);
            return -1;
        }
        memcpy(arr + curpos_byte, bins[i].elements, bins[i].size * elsize);
        curpos_byte += bins[i].size * elsize;
        PyMem_RawFree(bins[i].elements);
    }

    PyMem_RawFree(bins);
    PyMem_RawFree(sampleTree);
    PyMem_RawFree(targetbin_indices);
    PyMem_RawFree(slope);
    PyMem_RawFree(offset);
    return 0; // Success
}


// Main entry point for NumPy sort
// 'unused' is actually PyArrayObject *arr from which 'compare' and 'elsize' are derived.
int npy_tbsort(void *start, npy_intp num, void *arr_obj_unused) {
    PyArrayObject *arr_obj = (PyArrayObject *)arr_obj_unused;
    PyArray_CompareFunc *compare = PyArray_DESCR(arr_obj)->f->compare;
    int elsize = PyArray_DESCR(arr_obj)->elsize;

    if (num <= 1) {
        return 0;
    }
    if (compare == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot sort array data type with no comparison function");
        return -1;
    }
    if (elsize == 0) { // Should not happen for sortable types
        return 0;
    }

    // Seed random number generator - ideally this is done once at module init
    // or a NumPy specific RNG is used. For now, simple srand.
    // static int srand_called = 0;
    // if (!srand_called) {
    //    srand(time(NULL));
    //    srand_called = 1;
    // }
    // NumPy often avoids srand(time(NULL)) for determinism/testing.
    // If TBSort truly needs randomness, it should use NumPy's RNG facilities.
    // For now, we'll assume the quality of rand() is sufficient for pivot selection.

    return TBSort_generic((char *)start, num, compare, elsize, arr_obj);
}

// No main function here anymore. Original main is removed.
// Helper functions like printArray, original compare_integers are also removed or adapted.
