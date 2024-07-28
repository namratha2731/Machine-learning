def count_common_elements(list1, list2):
  
    set1 = set(list1)
    set2 = set(list2)

  
    common_elements = set1.intersection(set2)

    return len(common_elements)


list1 = list(map(int, input("Enter the first list of integers : ").split()))
list2 = list(map(int, input("Enter the second list of integers : ").split()))


common_count = count_common_elements(list1, list2)

print(f"Number of common elements: {common_count}")