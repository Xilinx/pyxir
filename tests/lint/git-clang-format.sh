

set -e
set -u
set -o pipefail

echo "Running git-clang-format against" HEAD~1
git-clang-format-11 --diff --extensions hpp,cpp HEAD~1 1> /tmp/$$.clang-format.txt
echo "---------clang-format log----------"
cat /tmp/$$.clang-format.txt
echo ""