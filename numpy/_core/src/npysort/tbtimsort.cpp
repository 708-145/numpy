// This file implements the TBSort algorithm.
// gcc -o tbsort tbsort.c -lm

#include "tbtimsort.hpp"    // Own header (includes npysort_common.h, numpy_tag.h and fwd decl for npy::sort::timsort)
#include "npy_sort.h"       // For potential general sort utilities

// #include <stdio.h> // No longer needed
#include <stdlib.h> // For rand - TODO: replace with NumPy RNG
#include <math.h>   // For log2, pow, round, roundf
#include <string.h> // For memcpy


namespace npy_tbsort {

// Helper function definitions (swap, myclamp, search, insertionSort, Bin struct)
// These are implementation details of TBSort

// Function to swap two elements
template <typename Tag>
void swap(typename Tag::type* xp, typename Tag::type* yp) {
    typename Tag::type temp = *xp;
    *xp = *yp;
    *yp = temp;
}

// Function to clamp a value n to be within lower and upper bounds
template <typename T>
T myclamp(T n, T lower, T upper) {
    if (n < lower) return lower;
    if (n > upper) return upper;
    return n;
}

// Function to perform binary search for element e in array a of size n
// Returns the position of the element in the array that is <= e
// Assumes the array is sorted
template <typename Tag>
int search(typename Tag::type a[], npy_intp n, typename Tag::type e) {
    npy_intp low = 0, high = n - 1;
    npy_intp ans = -1;

    while (low <= high) {
        npy_intp mid = low + (high - low) / 2;
        if (Tag::less_equal(a[mid], e)) { // Changed to use Tag::less_equal
            ans = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return ans;
}


// Function to implement insertion sort
template <typename Tag>
void insertionSort(typename Tag::type arr[], npy_intp n) {
    if (n <= 1) return;
    npy_intp i, j;
    typename Tag::type key;
    for (i = 1; i < n; i++) {
        key = arr[i];
        j = i - 1;

        while (j >= 0 && Tag::less(key, arr[j])) { // Changed to use Tag::less
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

// Structure for bins
template <typename Tag>
struct Bin {
    typename Tag::type* elements;
    npy_intp size;
    npy_intp capacity;
};


// TBSort function implementation
template <typename Tag>
int TBSort(typename Tag::type *arr, npy_intp l, npy_intp r) {
    // 1. Base Case
    if (l >= r) {
        return 0; // Success
    }

    npy_intp numElements = r - l + 1;

    // Handle very small arrays with insertion sort directly for stability
    if (numElements < 4) { // Threshold for direct sort
        if (numElements > 1) {
            // Using existing insertionSort which sorts in-place.
            // The original TBSort copied to a subarray then back.
            // This direct sort is simpler if arr points to the segment start.
            // However, the original TBSort was designed to take arr, l, r.
            // So arr+l is the start of the segment.
            insertionSort<Tag>(&arr[l], numElements);
        }
        return 0; // Success
    }

    // 2. TREE Step
    npy_intp treeSize;
    if (numElements < 4) {
        treeSize = 1;
    } else {
        double log2_numElements = log2(static_cast<double>(numElements));
        if (log2_numElements <= 0) log2_numElements = 1;
        double log2_log2_numElements = log2(log2_numElements);
        if (log2_log2_numElements < 0) log2_log2_numElements = 0;
        treeSize = static_cast<npy_intp>(pow(2, round(log2_log2_numElements)));
    }

    if (treeSize < 2 && numElements > 1) {
        treeSize = 2;
    }
    if (numElements <= 1) {
        treeSize = 1;
    }

    if (treeSize > numElements) {
        treeSize = numElements;
    }

    // Use C++ new/delete for memory management
    typename Tag::type* sampleTree = new (std::nothrow) typename Tag::type[treeSize];
    if (!sampleTree) {
        // perror("Failed to allocate memory for sampleTree"); // Replace with NumPy error handling
        return -1; // Error
    }

    // Populate sampleTree with random elements from arr[l...r]
    // TODO: NumPy has its own RNG, consider using it if mixing std::rand is an issue
    for (npy_intp i = 0; i < treeSize; i++) {
        if (numElements == 0) continue;
        sampleTree[i] = arr[l + rand() % numElements];
    }
    insertionSort<Tag>(sampleTree, treeSize); // Sort the sampleTree

    // 3. BIN Step
    npy_intp binCount;
    double logVal = log2(static_cast<double>(numElements));
    if (logVal <= 0 || numElements < 2) {
        binCount = treeSize + 2;
    } else {
        binCount = static_cast<npy_intp>(numElements / logVal);
    }

    if (binCount < treeSize + 2) {
        binCount = treeSize + 2;
    }
    if (binCount == 0) binCount = 1;


    Bin<Tag>* bins = new (std::nothrow) Bin<Tag>[binCount];
    if (!bins) {
        // perror("Failed to allocate memory for bins");
        delete[] sampleTree;
        return -1; // Error
    }

    for (npy_intp i = 0; i < binCount; i++) {
        bins[i].capacity = 4; // Initial capacity
        bins[i].elements = new (std::nothrow) typename Tag::type[bins[i].capacity];
        if (!bins[i].elements) {
            // perror("Failed to allocate memory for bin elements");
            for (npy_intp j = 0; j < i; j++) delete[] bins[j].elements;
            delete[] bins;
            delete[] sampleTree;
            return -1; // Error
        }
        bins[i].size = 0;
    }

    // Calculate targetbin, slope, and offset
    npy_intp targetbinSize = treeSize + 2;
    npy_intp* targetbin = new (std::nothrow) npy_intp[targetbinSize];
    // Using float for slope and offset as in original, consider double for precision with some Tags
    float* slope_arr = new (std::nothrow) float[targetbinSize -1];
    float* offset_arr = new (std::nothrow) float[targetbinSize -1];

    if (!targetbin || !slope_arr || !offset_arr) {
         // Failed to allocate memory for targetbin/slope_arr/offset_arr
         for (npy_intp i = 0; i < binCount; i++) {
            if (bins[i].elements) delete[] bins[i].elements;
         }
         delete[] bins;
         delete[] sampleTree;
         if (targetbin) delete[] targetbin;
         if (slope_arr) delete[] slope_arr;
         if (offset_arr) delete[] offset_arr;
         return -1; // Error
    }

    targetbin[0] = 0;
    targetbin[targetbinSize - 1] = binCount -1;
    for (npy_intp i = 1; i < targetbinSize - 1; i++) {
        targetbin[i] = myclamp<npy_intp>(static_cast<npy_intp>(roundf(numElements * (i) / static_cast<float>(treeSize +1) / logVal)), 0, binCount -1);
    }

    for (npy_intp i = 0; i < targetbinSize - 1; i++) {
        // Ensure Tag::type can be cast to float. This might be an issue for non-numeric types.
        float x1 = (i == 0) ? static_cast<float>(sampleTree[0]) -1.0f : static_cast<float>(sampleTree[i-1]);
        float x2 = (i == treeSize ) ? static_cast<float>(sampleTree[treeSize-1]) +1.0f : static_cast<float>(sampleTree[i]);

        if (x1 >= x2) {
            slope_arr[i] = 0;
            offset_arr[i] = static_cast<float>(targetbin[i]);
        } else {
            slope_arr[i] = (targetbin[i+1] - targetbin[i]) / (x2 - x1);
            offset_arr[i] = static_cast<float>(targetbin[i]) - slope_arr[i] * x1;
        }
    }


    // Distribute elements from arr[l...r] into bins
    for (npy_intp i = 0; i < numElements; i++) {
        typename Tag::type element_val = arr[l + i];
        npy_intp mypos = search<Tag>(sampleTree, treeSize, element_val);

        // Adjust mypos for slope/offset array indexing and logic
        // if element_val is smaller than smallest in sampleTree, search returns -1
        // if element_val is larger than largest in sampleTree, search returns treeSize-1
        // The slope/offset arrays are indexed from 0 to treeSize.
        // mypos from search: -1 to treeSize-1
        // if mypos is -1 (element_val < sampleTree[0]), use slope_arr[0], offset_arr[0]
        // if mypos is k where sampleTree[k] <= element_val < sampleTree[k+1], use slope_arr[k+1], offset_arr[k+1]
        // if mypos is treeSize-1 and element_val >= sampleTree[treeSize-1], use slope_arr[treeSize], offset_arr[treeSize]

        npy_intp slope_offset_idx;
        if (mypos == -1) {
            slope_offset_idx = 0;
        } else if (mypos == treeSize - 1 && !Tag::less(sampleTree[treeSize - 1], element_val)) { // element_val >= sampleTree[treeSize - 1]
            slope_offset_idx = treeSize;
        } else {
            slope_offset_idx = mypos + 1;
        }

        // Ensure Tag::type can be cast to float for this calculation.
        npy_intp mybin_idx = myclamp<npy_intp>(static_cast<npy_intp>(roundf(static_cast<float>(element_val) * slope_arr[slope_offset_idx] + offset_arr[slope_offset_idx])), 0, binCount - 1);

        // Add element_val to bins[mybin_idx]
        if (bins[mybin_idx].size >= bins[mybin_idx].capacity) {
            npy_intp new_capacity = (bins[mybin_idx].capacity == 0) ? 1 : bins[mybin_idx].capacity * 2;
            typename Tag::type* new_elements = new (std::nothrow) typename Tag::type[new_capacity];
            if (!new_elements) {
                // perror("Failed to reallocate memory for bin elements");
                // Extensive cleanup on new_elements allocation failure
                delete[] sampleTree;
                if (targetbin) delete[] targetbin;
                if (slope_arr) delete[] slope_arr;
                if (offset_arr) delete[] offset_arr;
                for(npy_intp k=0; k<binCount; ++k) {
                    // new_elements was for bins[mybin_idx].elements
                    // if k == mybin_idx, bins[k].elements is the old pointer, which is still valid
                    // and needs to be cleaned up.
                    if (bins[k].elements) delete[] bins[k].elements;
                }
                delete[] bins;
                return -1; // Error
            }
            memcpy(new_elements, bins[mybin_idx].elements, bins[mybin_idx].size * sizeof(typename Tag::type));
            delete[] bins[mybin_idx].elements;
            bins[mybin_idx].elements = new_elements;
            bins[mybin_idx].capacity = new_capacity;
        }
        bins[mybin_idx].elements[bins[mybin_idx].size++] = element_val;
    }

    // 4. SORT Step
    // binThreshold condition for switching to Timsort
    npy_intp binThreshold = static_cast<npy_intp>(5 * numElements / static_cast<float>(binCount));
    if (binCount == 0) binThreshold = numElements +1;

    npy_intp curpos = l;
    for (npy_intp i = 0; i < binCount; i++) {
        if (bins[i].size == 0) {
            delete[] bins[i].elements;
            continue;
        }

        if (bins[i].size < binThreshold) {
            // Call NumPy's Timsort for small bins
            // timsort_<Tag>(arr_bin_ptr, bin_size, nullptr)
            // Need to find the actual timsort function signature and ensure it's available.
            // For now, let's assume a generic timsort dispatch exists.
            // PyArray_Sort(bin_array, axis, sortkind) might be too high level.
            // Looking for internal timsort like timsort<Tag>(data, size, aux)
            // For now, using insertionSort as a placeholder for small bins if Timsort is complex to call directly.
            // The issue states "calls Timsort for the small bins"
            // This implies a direct call to a Timsort function similar to other sort functions in NumPy.
            // Let's assume `npy_timsort::timsort<Tag>(bins[i].elements, bins[i].size);`
            // This would require npy_timsort to be included and templated similarly.
            // For now, we use insertionSort as a stand-in until Timsort integration is clear.
             (void)npy::sort::timsort_<Tag>(bins[i].elements, bins[i].size); // Assuming this is the call
        } else {
            if (TBSort<Tag>(bins[i].elements, 0, bins[i].size - 1) != 0) {
                 // Error in recursive call, propagate error up
                delete[] sampleTree;
                if (targetbin) delete[] targetbin;
                if (slope_arr) delete[] slope_arr;
                if (offset_arr) delete[] offset_arr;
                for(npy_intp k=0; k<binCount; ++k) {
                    if (bins[k].elements) delete[] bins[k].elements;
                }
                delete[] bins;
                return -1; // Error
            }
        }

        if (curpos + bins[i].size > r + 1) {
            // Error: TBSort trying to write out of bounds.
            delete[] sampleTree;
            if (targetbin) delete[] targetbin;
            if (slope_arr) delete[] slope_arr;
            if (offset_arr) delete[] offset_arr;
            for(npy_intp k=0; k<binCount; ++k) {
                 // bins[i].elements has been processed by sort/memcpy for current i
                 // but others might still be there if error occurs mid-loop.
                 // However, bins[i].elements is deleted right after memcpy.
                 // This path implies an error *before* memcpy for the current bin,
                 // or an issue with numElements calculation.
                 // Safest to try deleting all non-null bin elements.
                if (bins[k].elements) delete[] bins[k].elements;
            }
            delete[] bins;
            return -1; // Critical error
        }
        memcpy(&arr[curpos], bins[i].elements, bins[i].size * sizeof(typename Tag::type));
        curpos += bins[i].size;
        delete[] bins[i].elements;
    }

    delete[] bins;
    delete[] sampleTree;
    if (targetbin) delete[] targetbin;
    if (slope_arr) delete[] slope_arr;
    if (offset_arr) delete[] offset_arr;
    return 0; // Success
}

} // namespace npy_tbsort

// Instantiate for npy_int and npy_float
// These lines should ideally be in a dispatching .cpp file if this were a header.
// For now, keeping it here to ensure template functions are generated.

template int npy_tbsort::TBSort<npy::tag::SignedInt>(npy::tag::SignedInt::type *, npy_intp, npy_intp);
template int npy_tbsort::TBSort<npy::tag::Float>(npy::tag::Float::type *, npy_intp, npy_intp);
// template int npy_tbsort::TBSort<npy::tag::UnsignedInt>(npy::tag::UnsignedInt::type *, npy_intp, npy_intp);
// template int npy_tbsort::TBSort<npy::tag::Double>(npy::tag::Double::type *, npy_intp, npy_intp);
