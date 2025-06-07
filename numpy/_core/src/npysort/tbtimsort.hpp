#ifndef NPYSORT_TBTIMSORT_HPP
#define NPYSORT_TBTIMSORT_HPP

#include "npysort_common.h" // For npy_intp
#include "numpy_tag.h"      // For npy::tag and Tag::type

// Forward declaration for the Timsort function used by TBSort for small bins.
// This assumes the actual implementation is available elsewhere (e.g., in timsort.cpp's machinery)
// and can be linked. The signature should match what TBSort calls.
namespace npy { namespace sort {
    // It seems timsort internal functions might be like:
    // template <typename Tag> int timsort_(void *start, npy_intp num)
    // or template <typename Tag> void timsort(typename Tag::type *, npy_intp, npy_intp*)
    // The call in tbtimsort.cpp is (void)npy::sort::timsort<Tag>(bins[i].elements, bins[i].size);
    // This should match the actual Timsort function that TBSort will call for small bins.
    // Based on timsort.cpp, this is likely timsort_<Tag>(void*, npy_intp)
    template <typename Tag>
    int timsort_(void *start, npy_intp num);
}} // namespace npy::sort


namespace npy_tbsort {

// Declaration of the TBSort function
template <typename Tag>
int TBSort(typename Tag::type *arr, npy_intp l, npy_intp r);

} // namespace npy_tbsort

#endif // NPYSORT_TBTIMSORT_HPP
