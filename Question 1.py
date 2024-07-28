def vowels_or_consonants(input_string):
    vowels = "aeiouAEIOU"
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    vowel_count = 0
    consonants_count = 0

    for char in input_string:
        if char in vowels:
            vowel_count += 1
        elif char in consonants:
            consonants_count += 1

    return vowel_count, consonants_count

input_string = input("Enter a string: ")

vowel_count, consonants_count = vowels_or_consonants(input_string)

print(f"Number of vowels: {vowel_count}")
print(f"Number of consonants: {consonants_count}")