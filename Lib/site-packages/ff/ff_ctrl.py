#!/usr/bin/python
"""
ff control
"""
import os, time, re
__version__ = 0.2

options = {}

HOME_PATH = os.environ["HOME"]
#BASHRC_PATH = HOME_PATH + '/.bashrc'
BASHRC_PATH = HOME_PATH + '/.bash_profile'
FFRC_PATH = HOME_PATH + '/.ffrc'

ACTION_INSTALL = 'install'
ACTION_HELP = 'help'

def printHelp():
  print 'Example:'
  print 'ff_ctrl action [--params ...]'
  print '''
      Actions:
        install : install ff script'''

def install():
  print 'Installing ff script...'
  #print 'Dry run:' + str(options.dryrun)

  bashrcExist = os.path.exists(BASHRC_PATH)
  ffrcExist = os.path.exists(FFRC_PATH)
  if not bashrcExist:
    print 'Bash profile does not exsit, creating it first...'
    printRun('touch ' + BASHRC_PATH)

  if ffrcExist:
    print '~/.ffrc already exist. Overriding...'
    createFFRC()
    appendFFRC2bash()
  else:
    print 'Creating ~/.ffrc'
    createFFRC()
    appendFFRC2bash()

def printRun(cmd):
  print cmd
  assert os.system(cmd) == 0

def writeContent(filePath, content):
  fd = open(filePath, "w");
  fd.write(content)
  fd.close()

def createFFRC():
  writeContent(FFRC_PATH, FFRC_CONTENT)
  printRun('. ~/.ffrc')

def appendFFRC2bash():
  fd = open(BASHRC_PATH, "r")
  lines = fd.readlines()
  hasFFRCappend = len([x for x in lines if 'ZHIFENG_FF_PLUGIN_START' in x or 'ZHIFENG_FF_PLUGIN_END' in x ]) == 2
  if hasFFRCappend:
    print 'FFRC append already exist'
  else:
    print 'Appending FFRC to bash profile'
    with open(BASHRC_PATH, "a") as myfile:
      myfile.write(FFRC_APPENDIX)

  print 'Setup completed, please reload bash profile.'

def main():
  from optparse import OptionParser
  parser = OptionParser()
  #parser.add_option('--dry', dest='dryrun', action='store_true', help='dry run')

  (opts,args) = parser.parse_args()
  global options
  options = opts
  print 'options:', options
  print 'args:', args
  print '\n\n'

  if (len(args) == 0):
    parser.print_help()
    printHelp()
    exit()

  action = args[0]
  # action that target a single app
  ALL_ACTION = {
             ACTION_INSTALL: install,
             ACTION_HELP: printHelp
  }

  if action in ALL_ACTION:
    ALL_ACTION[action](*args[1:])
    print ('\n\n')
    exit(0)
  else:
    printHelp()

FFRC_CONTENT = '''
#!/bin/bash
############################## ZHIFENG ffrc START #####################
# FFRC Version 0.1

if [ "$(uname)" = Darwin ]; then
  export __FF_OS_TYPE=MACOS
else
  export __FF_OS_TYPE=LINUX
fi

# Function key binding
if [ -z "$__FF__BIND_FUNC_KEY" ]; then
  echo
else
  if [ $__FF_OS_TYPE == MACOS ]; then
    bind '"\eOP":"vim $F1\015"'
    bind '"\eOQ":"vim $F2\015"'
    bind '"\eOR":"vim $F3\015"'
    bind '"\eOS":"vim $F4\015"'
    bind '"\e[15~":"vim $F5\015"'
    bind '"\e[17~":"vim $F6\015"'
    bind '"\e[18~":"vim $F7\015"'
    bind '"\e[19~":"vim $F8\015n"'
    bind '"\e[20~":"vim $F9\015"'
    bind '"\e[21~":"vim $F10\015"'
    __FF__BIND_FUNC_KEY=1
  else
    bind '"\eOP":"vim $F1\n"'
    bind '"\eOQ":"vim $F2\n"'
    bind '"\eOR":"vim $F3\n"'
    bind '"\eOS":"vim $F4\n"'
    bind '"\e[15~":"vim $F5\n"'
    bind '"\e[17~":"vim $F6\n"'
    bind '"\e[18~":"vim $F7\n"'
    bind '"\e[19~":"vim $F8\n"'
    bind '"\e[20~":"vim $F9\n"'
    bind '"\e[21~":"vim $F10\n"'
    __FF__BIND_FUNC_KEY=1
  fi
fi

# Quick alias
alias rf1='load_file $F1'
alias rf2='load_file $F2'
alias rf3='load_file $F3'
alias rf4='load_file $F4'
alias rf5='load_file $F5'
alias rf6='load_file $F6'
alias rf7='load_file $F7'
alias rf8='load_file $F8'
alias rf9='load_file $F9'

alias v1='vim $F1'
alias v2='vim $F2'
alias v3='vim $F3'
alias v4='vim $F4'
alias v5='vim $F5'
alias v6='vim $F6'
alias v7='vim $F7'
alias v8='vim $F8'
alias v9='vim $F9'

# Main utt function
function source_file() {
  if [ $# -lt 1 ] ; then
    TARGET=.
  else
    TARGET=$1
  fi
  find $TARGET -type d \( -path "*/.svn" -o -path "*/build" -o -path "*/.git"  \) -prune -o \
  -type f \( -iname "tags" \) -prune -o -type f -size +100000c -prune -o -type f -exec grep -Il . {} \;
}

_mapff() {
  i=1
  export __all_ff=$* 
  for file in $* 
  do
    if ((i % 2 == 0))
    then
      printf "\e[1;32m"
    fi
    export F$i=$file
    echo "[F$i]" $file 
    let " i+= 1"
    printf "\e[0m"
  done
}

_ff() {
# \* escaping the asterisk avoid shell expanding
  find . -name \*"$1"\* -o -type d \( -path "*/build" -o -path "*/.svn" -o -path "*/zip" -o -path "*/.git" \) -prune | egrep -v "./build|.git|.svn|./zip"
}

_ffb() {
  find . -name \*"$1"\* -o -type d \( -path "*/.svn" -o -path "*/.git" \) -prune | egrep -v ".git|.svn"
}

ff() {
  if [ $# -lt 1 ] ; then
    echo "usage: ff pattern"
    return
  fi
  _mapff `_ff $*`
}

ffb() {
  if [ $# -lt 1 ] ; then
    echo "usage: ffb pattern"
    return
  fi
  _mapff `_ffb $*`
}

editall () {
  vim $__all_ff
}

function gg() {
  if [ $# -lt 1 ] ; then
    echo "usage: greplg keyword"
    return
  fi
  source_file | xargs -I "fn" grep "$1" -n -B 2 --with-filename -A 2 --color=auto "fn"
}

function ggg() {
  if [ $# -lt 1 ] ; then
    echo "usage: greplg keyword"
    return
  fi
  source_file $WP | xargs -I "fn" grep "$1" -n -B 2 --with-filename -A 2 --color=auto "fn"
}

function gglist() {
  if [ $# -lt 1 ] ; then
    echo "usage: greplg keyword"
    return
  fi
  _mapff `source_file | xargs -I "fn" grep "$1" --with-filename "fn" | cut -d: -f 1 | uniq`
}

gff() {
  if [ $# -lt 1 ] ; then
    echo "usage: ff filepattern"
    return
  fi
  _mapff `find $WP -name \*"$1"\* -o -type d \( -path "*/build" -o -p "*/zip" -o -path "*/.svn" -o -path "*/.git" \) -prune | egrep -v "./build|.git|.svn|./zip" | grep --color $1`
}

load_file() {
  if [ $# -lt 1 ] ; then
    echo "load_file F1 or 1 or filename"
    return
  else
    vim -c ":qa" $1
  fi
}
############################## ZHIFENG ffrc END #####################
'''

FFRC_APPENDIX = '''
########## ZHIFENG_FF_PLUGIN_START #############
if [ -f ~/.ffrc ]; then
  . ~/.ffrc
fi
########## ZHIFENG_FF_PLUGIN_END ##############
'''

if __name__ == '__main__':
  main()
