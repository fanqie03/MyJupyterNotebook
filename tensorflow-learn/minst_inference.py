import tensorflow as tf

# 定义前向传播过程以及神经网络中的参数。
# 定义神经网络结构相关参数。
INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500

# 通过tf.get_variable函数来获取变量。在训练神经网络时会创建这些变量；
# 在测试时会用过保存的模型加载这些变量。而且更加方便的是，因为可以在
# 变量加载时将滑动平均变量重命名，所以可以直接通过同样的名字在训练时
# 将
# c初始化变量，若有正则化不为空，则将正则化加入损失集合中
def get_weight_variable(shape,regularizer):
    '''
    c初始化变量，若有正则化不为空，则将正则化加入损失集合中
    '''
    weights = tf.get_variable(
        'weights',shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

# 定义神经网络的前向传播过程
def inference(input_tensor, regularizer):
    # 声明第一层变量并完成前向传播过程。
    with tf.variable_scope('layer'):
        weights = get_weight_variable([INPUT_NODE],[LAYER1_NODE],regularizer)
        biases = tf.get_variable('biases',[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
        
    # 类似
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable('biases',[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
    
    return layer2


