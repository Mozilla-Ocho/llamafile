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
       three
- one

***

more text <http://google.com> link

word
  word

    needs blank line
normal

---

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

1. **`#include <iostream>`:** This line includes the `iostream` library,
   which provides input and output functionalities in C++. It allows us
   to use objects like `std::cout` for printing to the console.

2. **`int main() { ... }`:** This defines the `main` function, which is
   the entry point of any C++ program. The code inside the curly braces
   `{}` will be executed when the program runs.

3. **`int age = 25;`:** This line declares an integer variable named
   `age` and initializes it with the value 25.

4. **`std::cout << "Hello, world! My name is AI, and I am " << age << "
   years old." << std::endl;`:** This line prints a message to the
   console. Let's break it down:
   - `std::cout` is the standard output stream object.
   - `<<` is the insertion operator, used to send data to the output
     stream.
   - `"Hello, world! My name is AI, and I am "` is a string literal that
     will be printed as is.
   - `age` is the variable we declared earlier. Its value (25) will be
     inserted into the output stream.
   - `" years old."` is another string literal.
   - `std::endl` inserts a newline character, moving the cursor to the
     next line for any subsequent output.

5. **`return 0;`:** This line indicates that the program executed
   successfully. Returning 0 from the `main` function is a convention in
   C++ to signal successful execution.

**In summary:** This code snippet declares a variable, assigns it a
value, and then prints a message to the console including the variable's
value. It demonstrates basic C++ syntax for variable declaration,
output, and program structure.

---

Sure! Let's write a simple FORTRAN66 program that calculates the
factorial of a given number. I'll also provide an explanation of each
part of the code.

```fortran
      PROGRAM FACTORIAL
      INTEGER N, I, FACT
      PRINT *, 'Enter a positive integer: '
      READ *, N

      IF (N < 0) THEN
         PRINT *, 'Factorial is not defined for negative numbers.'
      ELSE
         FACT = 1
         DO 10 I = 2, N
10       FACT = FACT * I
         PRINT *, 'Factorial of ', N, ' is ', FACT
      END IF

      STOP
      END
```

### Explanation:

1. **Program Declaration:**
   ```fortran
   PROGRAM FACTORIAL
   ```
   This line declares the name of the program, which is `FACTORIAL`.

2. **Variable Declarations:**
   ```fortran
   INTEGER N, I, FACT
   ```
   This line declares three integer variables: `N` for the input number,
   `I` for the loop counter, and `FACT` for the factorial result.

3. **User Input:**
   ```fortran
   PRINT *, 'Enter a positive integer: '
   READ *, N
   ```
   - `PRINT *, 'Enter a positive integer: '` prints a prompt asking the
     user to enter a positive integer.
   - `READ *, N` reads the integer input from the user and stores it in
     the variable `N`.

4. **Condition to Check for Negative Input:**
   ```fortran
   IF (N < 0) THEN
      PRINT *, 'Factorial is not defined for negative numbers.'
   ELSE
      FACT = 1
      DO 10 I = 2, N
10    FACT = FACT * I
      PRINT *, 'Factorial of ', N, ' is ', FACT
   END IF
   ```
   - `IF (N < 0) THEN` checks if the input number is negative.
   - If `N` is negative, it prints a message indicating that the
     factorial is not defined for negative numbers.
   - If `N` is non-negative, it initializes `FACT` to 1 and uses a `DO`
     loop to calculate the factorial by multiplying `FACT` by each
     number from 2 to `N`.
   - Finally, it prints the factorial result.

5. **Stop Statement:**
   ```fortran
   STOP
   ```
   This line stops the execution of the program.

6. **End of Program:**
   ```fortran
   END
   ```
   This line marks the end of the program.

This program is a simple example of how to use variables, input/output
operations, conditional statements, and loops in FORTRAN66. It
demonstrates basic input/output operations and error handling.

## Rubby

```ruby
# comment
42
"String with #{ :interpolation }"
/regex$/
$/
[?😀, ?\ , ?', ?(] ?'a':'b'
"%d %d %d"%[1,2,3]
% 5 #

/
$$#$/
/omix

puts <<HERE<<<<THERE
foo 42
HERE
bla 43
THERE

puts "This is #{<<HERE.strip} evil"
incredibly
HERE

def `
  `hello` / /regex/
end

"hi #$" there" a
"hi #$abc there" a
"hi #{hi} there" a
/hi #$" there/ a
/hi #$abc there/ a
/hi #{hi} there/ a
_=;"#$q"

# Examples:
%q(Don't worry)            # => "Don't worry"
%Q(Today is #{Date.today}) # => "Today is 2024-11-08"
%r{^/path/to/\w+}          # => /^\/path\/to\/\w+/
%s(my_symbol)              # => :my_symbol
%w(foo bar baz)            # => ["foo", "bar", "baz"]
%W(foo #{1+1} baz)         # => ["foo", "2", "baz"]
%x(ls -l)                  # Executes `ls -l` and returns output
%i(foo bar baz)            # => [:foo, :bar, :baz]
%I(foo#{1+1} bar baz)      # => [:foo2, :bar, :baz]
```
