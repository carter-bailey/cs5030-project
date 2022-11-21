#include "utils.hpp"

int main()
{
    auto data = getCSV();
    for (auto d : data)
    {
        std::cout << d.genre << "\n";
    }
}
