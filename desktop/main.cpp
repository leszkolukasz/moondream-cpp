#include <iostream>
#include <moondream/moondream.h>

int main() {
  using namespace moondream;

  Moondream dream("./data");
  auto res = dream.caption("./frieren.jpg", "normal", 1000);

  std::cout << "Caption: " << res << std::endl;

  return 0;
}
