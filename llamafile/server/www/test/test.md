1. hi
 - there

bold: **bold**  
italic: *italic*  
strike: ~~strike~~  
bold+italic: ***bold***  
italic→bold: *hi***there**  
bold→italic: **hi***there*  
italic+strike: *~~strike~~*  
bold+strike: **~~strike~~**  
bold+italic+strike: ***~~strike~~***  
code: `plain`, *`italic`*, **`bold`**  
code+tick: ```left``, ``right```  
link: [link **bold** and *italic* stuff](https://google.com)  
escapable: \! \# \( \)\* \+ \- \. \< \> \[ \\ \[ \] \_ \` \{ \| \}  
non-escapable: \a \0 \Z \~  

* one
 * two
     - three
- one

¶1
lorem ipsum
dolor sit amet

¶2
consectetur adipiscing elit
sed do eiusmod tempor incididunt

***

more text <http://google.com> link

***

```cpp
#include <iostream>

int main() {
  // Declare a variable to store an integer
  int age = 25;

  // Print a message to the console
  std::cout << "Hello, world! My name is AI, and I am " << age << " years old." << std::endl;

  return 0;
}
```

**Explanation:**

1. **`#include <iostream>`:** This line includes the `iostream` library, which provides input and output functionalities in C++. It allows us to use objects like `std::cout` for printing to the console.

2. **`int main() { ... }`:** This defines the `main` function, which is the entry point of any C++ program. The code inside the curly braces `{}` will be executed when the program runs.

3. **`int age = 25;`:** This line declares an integer variable named `age` and initializes it with the value 25.

4. **`std::cout << "Hello, world! My name is AI, and I am " << age << " years old." << std::endl;`:** This line prints a message to the console. Let's break it down:
   - `std::cout` is the standard output stream object.
   - `<<` is the insertion operator, used to send data to the output stream.
   - `"Hello, world! My name is AI, and I am "` is a string literal that will be printed as is.
   - `age` is the variable we declared earlier. Its value (25) will be inserted into the output stream.
   - `" years old."` is another string literal.
   - `std::endl` inserts a newline character, moving the cursor to the next line for any subsequent output.

5. **`return 0;`:** This line indicates that the program executed successfully. Returning 0 from the `main` function is a convention in C++ to signal successful execution.

**In summary:** This code snippet declares a variable, assigns it a value, and then prints a message to the console including the variable's value. It demonstrates basic C++ syntax for variable declaration, output, and program structure.
