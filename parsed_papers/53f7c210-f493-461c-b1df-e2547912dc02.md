## Module 1 Cheat Sheet - Introduction to Linux

## Linux terminal tips

Use tab completion to autocomplete pathnames and command names.

Scroll through your command history with the Up Arrow and Down Arrow keys to find and re-run a command you already used.

## Getting information

Display the reference manual for the ls command:

None

man ls

## Browsing and navigating directories

## Special paths

| Symbol   | Represents path to                  |
|----------|-------------------------------------|
| ~        | home directory                      |
| /        | root directory                      |
| .        | present working directory           |
| ..       | parent of present working directory |

List files and directories in the current directory:

None

None

None

None

None

None

ls

## List files and directories in a directory:

ls path\_to\_directory

Return path to present working directory:

pwd

Change the current directory to a subdirectory:

cd child\_directory\_name

Tip: Because cd looks in the current directory for child\_directory\_name, you don't need to type the entire path.

## Change the current directory:

Up one level:

cd ../

To home:

cd ~


<!-- PAGE_BREAK -->



<!-- PAGE_BREAK -->


or None cd

## To some other directory:

```
None
```

cd path\_to\_directory

## Change the current directory to another one at the same level:

Suppose you have two sibling directories within the same directory, dir\_1 and dir\_2, and your present working directory is dir\_1. To switch to dir\_2, enter:

```
None
```

```
cd ../dir_2
```

Tip: Using .., you don't need to know the path to the parent directory to switch to a sibling.

## Change the current directory back to the directory you were in previously:

```
None cd -
```

## Upgrading and installing packages

Fetch and display up-to-date information about all upgradable packages:

```
None
```

```
sudo apt update
```


<!-- PAGE_BREAK -->


Upgrade to the latest supported version of nano:

None

```
None
```

sudo apt upgrade nano

## Install Vim:

sudo apt install vim

## Creating and editing files

Create a new text file and open it with nano:

```
None
```

nano file\_name.txt

Tip: If the file already exists, nano simply opens it for editing.

Next critical test: upload an updated version

1. Modify the PDF content slightly.

This is the most important test for our version-management architecture.

2. Export it as a new PDF.
4. Confirm that it resolves to the same document ID.
3. Upload it using the same authenticated account.
5. Confirm that MongoDB creates version 2.
7. Confirm that version 2 becomes active and version 1 is superseded.
6. Confirm that all version 2 chunks are indexed before activation.
8. Confirm that version 1's Qdrant points are deleted.

Expected result: