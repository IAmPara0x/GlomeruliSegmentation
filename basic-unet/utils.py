
def calc_equal_conv_padding(input_dim, kernel_size, dilation=1, stride=1):
  """
  calculates padding such that input_dim and output_dim are equal.
  Important for architectures like U-net.
  """
  return (stride*(input_dim - 1) + 1 + dilation * (kernel_size - 1) - input_dim) / 2

