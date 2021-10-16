import tensorflow as tf
'''
 常用函数 对应元素的四则运算
实现两个张量的对应元素相加 tf.add (张量1，张量2)
实现两个张量的对应元素相减 tf.subtract (张量1，张量2)
实现两个张量的对应元素相乘 tf.multiply (张量1，张量2)
实现两个张量的对应元素相除 tf.divide (张量1，张量2)
只有维度相同的张量才可以做四则运算

'''
a = tf.ones([1, 3])
b = tf.fill([1, 3], 3.)

print("a:", a)
print("b:", b)
print("a+b:", tf.add(a, b))
print("a-b:", tf.subtract(a, b))
print("a*b:", tf.multiply(a, b))
print("b/a:", tf.divide(b, a))
c=a+b
c1=a-b
c2=a*b
c3=a/b
c4=b//a
c5=b%a
c6=tf.math.log(a)#对数运算
c7=tf.exp(a)#指数运算

'''
计算某个张量的平方 tf.square (张量名) 􏰀
计算某个张量的n次方 tf.pow (张量名，n次方数) 􏰀
计算某个张量的开方 tf.sqrt (张量名)
'''
a1 = tf.fill([1, 2], 3.)
print("a:", a)
print("a的平方:", tf.pow(a1, 3))
print("a的平方:", tf.square(a1))
print("a的开方:", tf.sqrt(a1))

'''
实现两个矩阵的相乘
tf.matmul(矩阵1，矩阵2)
'''
a2 = tf.ones([3, 2])
b2 = tf.fill([2, 3], 3.)
print("a:", a2)
print("b:", b2)
print("a*b:", tf.matmul(a2, b2))
c8=a2@b2
print(c8)

a3 = tf.ones([4,3,2])
b3 = tf.fill([4,2,4],3.)
c9=a3@b3#后两维进行矩阵乘法
c10=tf.matmul(a3,b3)
print(c9,'\n',c10)

a3 = tf.ones([4,3,2])
b3 = tf.fill([2,4],3.)
b4=tf.broadcast_to(b3,[4,2,4])
c11=a3@b4



