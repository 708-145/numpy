#ifndef NPYSORT_TBSTABLESORT_HPP
#define NPYSORT_TBSTABLESORT_HPP

#include "npysort_common.h" // For npy_intp
#include "numpy_tag.h"      // For npy::tag and Tag::type

// Forward declaration for the Mergesort function used by TBStablesort for small bins.
// This assumes the actual implementation is available elsewhere (e.g., in mergesort.cpp)
// and can be linked. The signature should match what TBStablesort calls.
namespace npy { namespace sort {
    // Assuming mergesort internal function is like:
    // template <typename Tag> int mergesort_(typename Tag::type *start, npy_intp num);
    // (Note: Original mergesort in NumPy might have a different signature or might not be templated with Tag directly in this way for internal calls)
    // For now, we'll assume a compatible templated version exists or can be made.
    template <typename Tag>
    int mergesort_(typename Tag::type *start, npy_intp num);
}} // namespace npy::sort


namespace npy_tbstablesort {

// Declaration of the TBStablesort function
template <typename Tag>
int TBstablesort(typename Tag::type *arr, npy_intp l, npy_intp r);

} // namespace npy_tbstablesort

#endif // NPYSORT_TBSTABLESORT_HPP
