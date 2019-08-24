# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.datasets import make_blobs


features,_ = make_blobs(n_samples=1000, n_features=2, random_state=1)

True_val = features[0,0]
features[0,0] = np.nan
mean_imputer = Imputer(strategy="median", axis = 0)

featured_mean_imp = mean_imputer.fit_transform(features)

print("ture value:", True_val)
print("Imputed_val: ", featured_mean_imp[0,0])
##########################################################################

def IsPalin(str):
    if(str==str[::-1]):
        print(str," is a palindrome")
    else:
        print(str," is not a palindrome")

strings = "A man a plan a canal Panama"   
strings= strings.replace(" ","")     
IsPalin(strings.lower())
##########################################################################TwoSUm variation 1
def twoSum(arr,target):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if(arr[i]+arr[j+1] == target):
                print(arr[i], " + ", arr[j+1], " is equals to ",target)
                print("index", [i,j+1])
            else:
                print("no match found for target value in the given array")
#twoSum([2,3,4,1,2],5)
##########################################################################TwoSUm variation 2
                
\
def twoSumHash(arr,target):
    hashtemp = {}
    for i, num in enumerate(arr): # 0 , 2
        if target - num in hashtemp: # 5-2 in hastemp
            print(hashtemp[target-num],i)
        hashtemp[num] = i
                
twoSumHash(nums,7)
#########################################################################TwoSUm variation 3
#pair sum
def pairsum(nums,target):
    if(len(nums) ,2):
        print("too small array")
        
    countSet = {}
    for i, num in enumerate(nums):
        if target - num in countSet:
            print(countSet[target-num],num)
            
        countSet[num] = i
pairsum(nums,7)
##########################################################################Longest String
a = "dakitikisa"  # Provide any string

def LongestSubstring(str):
    count=[]
    for j in range(len(a)):
        for i in range(j,len(a)):
            if a[j:i+1] == a[i:j-1:-1]:      
                count.append(i+1-j)
                print
    print("Maximum size of Palindrome within String is :", max(count)) 

LongestSubstring(a)
######################################################################Largest Sum
nums = [7,1,2,-1,3,4,10,-12,3,21,-19]

def LargestSum(nums):
    if len(nums) == 0:
        return print("too small of an array")
    
    Max_sum = current_sum = nums[0]
    for i in nums[1:]:
        current_sum = max(current_sum+i,i)
        Max_sum = max(current_sum,Max_sum)
        
    print("Max Sum is ",Max_sum)

LargestSum(nums)
######################################################################Anagram
#Anagram = ""
def anagrm(s1,s2):
    s1 = s1.replace(' ','').lower()
    s2 = s2.replace(' ','').lower()
    
    if(len(s1) != len(s2)):
        return False
    
    dictionary = {}
    for letter in s1:
        if(letter in dictionary):
            dictionary[letter] +=1
        else:
            dictionary[letter] = 1
            
    for letter in s2:
        if(letter in dictionary):
            dictionary[letter] -=1
        else:
            dictionay =1
    
    for k in dictionary:
        if dictionary[k] != 0:
            return False
    return True
 
%timeit(anagrm('dam', 'mad'))  
#####################################################################String Reverse
reverse = "hi what up!"
reverse[::-1]
def ReverseString(strng):
    n[] = 0
    for i in range(len(strng)):
        n[i] = strng[::-1]
        
    print(n[i])
    
%timeit(ReverseString(reverse))
# Reserve a string by each word######################################
start = "This is the best"
end = "Best the is this"
def reverse(start):
    return " ".join(reversed(start.split()))

%timeit(reverse(start))
#####################################################################String Reverse 
def Reverse(s):
    
    length = len(s)
    i = 0
    space = [' ']
    words = []
    while i < length:
        if s[i] not in space:
            start_Index = i
            
            while i < length and s[i] not in space:
                i+=1
            words.append(s[start_Index:i])
            
        i+=1
    return "".join(reversed(s))

print(Reverse("this is strange"))
###################################################################### common elements in an array

def common_elemtns(l1,l2):
    p1 = 0
    p2 = 0
    
    result = []
    while p1<len(l1) and p2 < len(l2):
        if(l1[p1] == l2[p2]):
            result.append(l1[p1])
            p1 +=1
            p2 +=2
        elif(l1[p1] > l2[p2]):
            p2 +=1
        else:
            p1 +=1
            
    return result

common_elemtns([1,3,4,6,7,9],[1,2,4,5,9,10])

#######################################################################mine sweeper
num_cols = 3
num_rows = 3
def Minesweeper(bombs, rows,cols):
    field = [[0 for i in range(num_cols)] for j in range(num_rows)]
    
    for bomb_location in bombs:
        (bomb_row,bomb_col) = bomb_location
        field[bomb_row, bomb_col] = -1
        
        row_range = range(bomb_row - 1, bomb_row +2)
        col_range = range(bomb_col -1 , bomb_col +2)
        
        for i in row_range:
            current_i = i
            for j in col_range:
                current_j = j


################################################################################ most frequent element in the list
def most_freq_eleme(li):
    
    dictionary = {}
    count = 0
    max_item = None
    
    for i in li:
        if i not in dictionary:
            dictionary[i] = 1
        else:
            dictionary[i] +=1
            
        if dictionary[i] > count:
            count = dictionary[i]
            max_item = i
            
     
    return(max_item)
    
print("most frequent number in the list is ",most_freq_eleme([3,2,1,2,2,3,3,3]))
##############################################################################Unique Chars


def UniqueChars(strings):
    
    s = strings.replace(' ','')
    
    chars = set()
    for i in s:
        if i in chars:
            print("inside if before else ",chars)
            return False
        else:
            chars.add(i)
    print(chars)
    return True

print(UniqueChars('axbe d'))
    

print(UniqueChars("abcdefe"))

###################################################################### Non repeat chars in a string

def Nonrepeats(s):
    
    s = s.replace(' ','').lower()
    
    char_count = {}
    uniq_chars = {}

    for c in s:
        if c not in char_count:
            char_count[c] =1
        else:
            char_count[c] +=1
    #return char_count
    
    print(c for c in s if char_count[c] == 1)

print(Nonrepeats('I like'))

################################################################## remove vowels from a string

def removeVowel(strng):
    
    strng = strng.replace(' ','')
    vowels = {'a','e','i','o','u'}
    
    for char in strng.lower():
        if char in vowels:
            strng = strng.replace(char,'')
            
    return strng        

print(removeVowel('hello there miss'))

##################################################################### defanging Ipaddress
Input: address = "1.1.1.1"
Output: "1[.]1[.]1[.]1"

def defangIP(strng):
    
    for i in strng:
        if i == '.':
            strng = strng.replace('.','[.]')
            
    return strng
defangIP('12.34.123')

###################################################################3 jewels and stones
Input: J = "aA", S = "aAbbbb"
Output: 3

def JewelsandStones(J,S):
    count= 0    
    J = J.replace(' ','')
    for s in S:
        if s in J:
            count +=1
    return count
    
JewelsandStones('aA','aAbbbb')

###################################################################