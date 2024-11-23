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

## Code

~~~
I can't believe it's not butter!
~~~

## Tabs

Tabs can make code blocks.

	code block

Tabs round up to 4 when mixed with spaces.

  	code block

- list item

		code block under li

## Emphasis

***strong emph***  
***strong** in emph*  
***emph* in strong**  
**in strong *emph***  
*in emph **strong***

The following patterns are less widely supported, but the intent is
clear and they are useful (especially in contexts like bibliography
entries):

*emph *with emph* in it*  
**strong **with strong** in it**

Many implementations have also restricted intraword emphasis to
the `*` forms, to avoid unwanted emphasis in words containing
internal underscores.  (It is best practice to put these in code
spans, but users often do not.)

internal emphasis: foo*bar*baz  
no emphasis: foo_bar_baz

## Backslash escapes

Any ASCII punctuation character may be backslash-escaped:

\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~

```````````````````````````````` example
<p>!&quot;#$%&amp;'()*+,-./:;&lt;=&gt;?@[\]^_`{|}~</p>
````````````````````````````````

Backslashes before other characters are treated as literal
backslashes:

\→\A\a\ \3\φ\«

```````````````````````````````` example
<p>\→\A\a\ \3\φ\«</p>
````````````````````````````````

Escaped characters are treated as regular characters and do
not have their usual Markdown meanings:

\*not emphasized\*  
\<br/> not a tag  
\[not a link](/foo)  
\`not code\`  
1\. not a list  
\* not a list  
\# not a heading  
\[foo]: /url "not a reference"  
\&ouml; not a character entity

```````````````````````````````` example
<p>*not emphasized*
&lt;br/&gt; not a tag
[not a link](/foo)
`not code`
1. not a list
* not a list
# not a heading
[foo]: /url &quot;not a reference&quot;
&amp;ouml; not a character entity</p>
````````````````````````````````

A backslash at the end of the line is a [hard line break]:

foo\
bar

```````````````````````````````` example
foo\
bar
.
<p>foo<br />
bar</p>
````````````````````````````````

## Rubby

```ruby
# This is a single-line comment

bone
c = p / r
puts puts
%q#hi#
module RDoc::Parser::RubyTools

Bundler.ui.info <<-EOS.gsub(/^ {8}/, "")
--- ERROR REPORT TEMPLATE -------------------------------------------------------
# Error Report
EOS

=begin ho
This is a
multi-line comment
=end yo

bone

# String literals
single_quoted = 'Hello, Ruby!'
double_quoted = "Hello, Ruby!"

# String interpolation (only works with double quotes)
name = "Alice"
greeting = "Hello, #{name}!"

# Heredoc syntax
long_text = <<-EOT
  This is a long piece of text.
  It can span multiple lines.
  Indentation is preserved.
EOT

# Flexible quoting
ha = %
ha = %q
flexible_quote = %q{This is a single-quoted string}
another_flexible = %Q{This is a double-quoted string with #{name}}

# String methods
upcase_string = "hello".upcase
downcase_string = "WORLD".downcase
capitalize_string = "ruby".capitalize

# String concatenation
concat_string = "Hello " + "World"
concat_with_shovel = "Hello " << "Ruby"

# Symbols (immutable strings, often used as hash keys)
symbol = :my_symbol

def is_prime?(num)
  return false if num <= 1
  (2..Math.sqrt(num)).each do |i|
    return false if num % i == 0
  end
  true
end

count = 0
num = 2
num = ?h
dog = :!hi

while count < 10
  if is_prime?(num)
    puts num
    count += 1
  end
  num += 1
end

puts /hi/
dog /hi/

exit
dog = $ h $$ h $@ h $! h $/ h $_ h $= e @ e $\ e $` e $< e $> e $' e
dog = @hi
dog = @@hi
dog = $hi ${hi}
dog = "h\"i"
dog = 'h\'i'
dog = `hi\`hi`

print <<'woffle end'
Hello, world!
woffle end

print <<''
Hello, world!

print <<END;
Hello, world!
END

print <<ez
Hello, world!
ez

print <<"end"
Hello, world!
end

print <<`CODE`
Hello, world!
CODE

print <<END1, <<END2;
First heredoc
END1
Second heredoc
END2

	print <<-EOF
	hello
	there
	EOF

	print <<-'EOF'
	hello
	there
	EOF

print "hi"

require 'hi'

print $# hi

if @yaml_config
  require 'yaml'
  cfg = YAML.load_file(HERE + @yaml_config)[:generate_module]
  @path_src     = cfg[:defaults][:path_src]   if nil?
  @path_inc     = cfg[:defaults][:path_inc]   if @path_inc.nil?
  @path_tst     = cfg[:defaults][:path_tst]   if @path_tst.nil?
  @update_svn   = cfg[:defaults][:update_svn] if @update_svn.nil?
  @extra_inc    = cfg[:includes]
  @boilerplates = cfg[:boilerplates]
else
  @boilerplates = {}
end

ugh %q(ugh (hi) ugh) ugh
ugh %q!ugh hi ugh! ugh
ugh %q#ugh hi ugh# ugh
ugh %q<ugh hi ugh> ugh
ugh %/augh hi ugh/ ugh

/hi/i

if line =~ /^((?:\s*TEST_\/CASE\s*\(.*?\)\s*)*)\s*void\s+((?:#{@options[:test_prefix]}).*)\s*\(\s*(.*)\s*\)/i
  arguments = $1
  name = $2
  call = $3
  args = nil
  if (@options[:use_param_tests] and !arguments.empty?)
    args = []
    arguments.scan(/\s*TEST_CASE\s*\((.*)\)\s*$/) {|a| args << a[0]}
  end
  tests_and_line_numbers << { :test => name, :args => args, :call => call, :line_number => 0 }
end

    mk << %{
SHELL = /bin/sh

# V=0 quiet, V=1 verbose.  other values don't work.
V = 0
Q1 = $(V:1=)
Q = $(Q1:0=@)
ECHO1 = $(V:1=@ #{CONFIG['NULLCMD']})
ECHO = $(ECHO1:0=@ echo)
NULLCMD = #{CONFIG['NULLCMD']}

#### Start of system configuration section. ####
#{"top_srcdir = " + $top_srcdir.sub(%r"\A#{Regexp.quote($topdir)}/", "$(topdir)/") if $extmk}
srcdir = #{srcdir.gsub(/\$\((srcdir)\)|\$\{(srcdir)\}/) {mkintpath(CONFIG[$1||$2]).unspace}}
topdir = #{mkintpath(topdir = $extmk ? CONFIG["topdir"] : $topdir).unspace}
hdrdir = #{(hdrdir = CONFIG["hdrdir"]) == topdir ? "$(topdir)" : mkintpath(hdrdir).unspace}
arch_hdrdir = #{$arch_hdrdir.quote}
PATH_SEPARATOR = #{CONFIG['PATH_SEPARATOR']}
VPATH = #{vpath.join(CONFIG['PATH_SEPARATOR'])}
}

class SimpleScanner < Scanner # :nodoc:
  def scan
    stag_reg = (stags == DEFAULT_STAGS) ? /(.*?)(<%[%=#]?|\z)/m : /(.*?)(#{stags.join('|')}|\z)/m
    etag_reg = (etags == DEFAULT_ETAGS) ? /(.*?)(%%?>|\z)/m : /(.*?)(#{etags.join('|')}|\z)/m
    scanner = StringScanner.new(@src)
    while ! scanner.eos?
      scanner.scan(@stag ? etag_reg : stag_reg)
      yield(scanner[1])
      yield(scanner[2])
    end
  end
end

  ARGV.reject! do |arg|
    case(arg)
      when '-cexception'
        options[:plugins] = [:cexception]; true
      when /\.*\.ya?ml/

        options = UnityTestRunnerGenerator.grab_config(arg); true
      when /\.*\.h/
        options[:includes] <<arg; true
      when /--(\w+)=\"?(.*)\"?/
        options[$1.to_sym] = $2; true
      else false
    end
  end
```

# D

```d
// d stands for dead, for
// what is dead may never die
// but rises again, harder and stronger
/* com /* com */ plain
/+ com /+ com +/ com +/ plain
/+/++/+/ plain
/+/++/++/ plain
/+//++/++/ plain
/+//++/+++/ plain
"str\"str" plain
`str"\str` plain
r"str\str" plain
q"/str/warn/" plain
q"<str>warn>" plain
q"(str"(str)"str)" plain
q"[str"[str]"str]" plain
q"eos
hello
eos" plain
q"eos!warn
warn
eos" plain
q"eos
warn
eos
warn
eos" plain
x"dead beef" plain
x"dead warn" plain
```
