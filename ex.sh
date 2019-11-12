echo "Movie Lens 100K :"
(cat uti1.txt) > input.txt 
# python3 test.py
g++ -fopenmp ml-100k.cpp
./a.out u1.base u1.test

echo ""
echo "Movie Lens 1M :"
(cat uti2.txt) > input.txt 
# python3 test.py
g++ -fopenmp ml-1m.cpp
./a.out ratings.dat u2.test

echo ""
echo "Netflix 1200:"
python3 main.py


cat out.txt 