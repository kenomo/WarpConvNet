#!/bin/bash

{
    echo "## Directory Tree"
    echo "\`\`\`"
    git ls-files | tree --fromfile -F --dirsfirst
    echo "\`\`\`"
} > DIR_STRUCTURE.md
