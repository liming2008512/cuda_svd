#include <vector>

using namespace std;

template <class T>
void vector_clear(vector<T> &v)
{
	vector<T> free_vector;
	free_vector.swap(v);
}
