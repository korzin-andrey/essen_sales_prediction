def merge(nums1, m: int, nums2, n: int) -> None:
    """
    Do not return anything, modify nums1 in-place instead.
    """
    if len(nums2) == 0:
        return nums1
    iter_1, iter_2 = n-1, m+n-1
    iter_aux = n-1
    while iter_1 != iter_2:
        val = nums2[iter_aux]
        if nums1[iter_1] < val:
            nums1[iter_2] = val
            iter_aux -= 1
        else:
            nums1[iter_2] = nums1[iter_1]
            iter_1 -= 1
        iter_2 -= 1
    return nums1


if __name__ == '__main__':
    # print(merge([1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3))
    print(merge([0], 0, [1, 2, 3], 3))
