class Factorytest():
  @staticmethod
  def factory(numerator):
    def division(denominator):
      return numerator/denominator

    return division


if __name__ == '__main__':
  f = Factorytest.factory(10)
  print(f(2))
  print(f(5))