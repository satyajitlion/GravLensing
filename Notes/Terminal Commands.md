### Notes:

- **cd** - change directory
	- Generally, the name is assumed to be a *relative* to the current directory tree
	- if name starts with / then it is an *absolute* name in the overall directory tree
	- Typing "cd" with no arguments takes you to the ***home*** directory
	- Typing "cd .." takes you up one level in the directory tree
- **ls** - list contents of current directory
	- Typing "ls -l" gives a "long" listing with information about each file. 
- **pwd** - print the name of the current directory
-  **mkdir** - make new directory
-  **rmkdir** - remove a directory
- **cp** - copy files 
- **mv** - move or rename files 
- **rm** - remove files files
- **more** - displays the contents of a (text) file, starting from the top, with prompts to continue after each full window
- **cat** - similar to more but without the prompts
- **head** - shows the beginning of a file
- **tail** - shows the end of a file
- " ? " - denotes any single character
- " * " - denotes any string of characters (of any length)
- If you press tab while writing out a directory name, it finishes the job for you. (Usually faster approach to navigating through the terminal.)
- "Alias" - string that stands for some (usually longer) string. If there are commands that you use frequently, setting up aliases can simplify things. You can use aliases to prevent some mistakes as well.
	- Ex:
		- **alias rm** = '**rm -i**'
		- **alias mv** = '**mv -i**'
		- **alias cp** = '**cp -i**'
- The '**-i**' flag tells the terminal to ask you before doing anything destructive (if you accidentally try to rename a file that already exists)
- You can output aliases in a special file called ```.bash_profile``` so that the aliases will automatically be activated when the terminal is opened.

##### tag: #Terminal #High-PerformanceComputing #Amarel