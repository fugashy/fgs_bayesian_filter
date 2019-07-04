# fgs_gaussian_filter

自己位置推定のためのガウシアンフィルタ群（になる予定）  
クラス設計を重視して，流れをつかみやすくすることを意識  
無駄にROS2パッケージなのは趣味

参考

- [確率ロボティクス](https://book.mynavi.jp/ec/products/detail/id=37337)

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)

  ```
  The MIT License (MIT)

  Copyright (c) 2016 Atsushi Sakai

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```

# How to use

- As ROS2

  ```bash
  ros2 run fgs_gaussian_filter filter_out
  ```

- As python3 script

  ```bash
  cd fgs_gaussian_filter
  ./ros2_entry_point
  ```
