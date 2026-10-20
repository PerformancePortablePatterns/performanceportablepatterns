---
icon: lucide/compass
---

# How the patterns are presented
This section gives an overview of how the design patterns are presented on this webpage.
The navigation section named "Patterns" in the navigational bar on the right hand side shows the name of the patterns.
Clicking on the pattern will open the subpage that contains the following sections describing the pattern.

## Synopsis

The synopsis is a concise summary of what a pattern does. It does not go into detail but allows the reader to quickly decide if a pattern matches a certain problem or not.
It is the first section of a pattern's subpage to allow quick scanning of the patterns.

## Mathematical expression

Many patterns can be described by a mathematical formula that they implement.
The advantage of this description is its precision while providing generality.
For example, a parallel scan can be expressed as:

The scan operation take a binary operator $\odot$ with identity I, and an ordered set $[a_0, a_1, ..., a_{n−1}]$ of $n$ elements, and returns the ordered set $[I, a_0, (a_0 \odot a_1), ..., (a_0 \odot a_1 \odot ... \odot a_{n−2})]$ [^1].

For $\odot=+$ this gives for element $i$:
$$
e_i = I + \sum_{k=0}^{i+1} a_k
$$


## Pseudocode

This section features a pseudo code implementation and supplementary description.
The main focus is to explain how the pattern works in detail without being specific to a certain language.
Nevertheless, the pseudocode is easy enough for people familiar with any programming language are able to read and understand the code.
Thus, it will not be complete or executable code but focusing on the easiest way to present the relevant information.

## Reference implementation

In this section, at least one reference implementation that is tested in the project's CI is implemented.
The reference implementation will feature unit testing and links to the code available on the github page.
This project strives to have multiple implementations in different languages and using various libraries if time allows.

## Known uses in open-source software

This last section will list known open-source libraries that use the respective pattern.
This provides users with links to see how the respective pattern is used in already existing software and provide more examples of what can be achieved with the respective pattern.

[^1]: Vector Models for Data-Parallel Computing, Guy E. Blelloch, MIT Press, 1990.
