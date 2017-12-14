
#coding:utf-8
import matplotlib.pyplot as plt
import ast
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号


filename = "Residual0.txt"
history = {}
with open(filename, "r") as f:
    history = ast.literal_eval(f.read())
fig = plt.figure(figsize=(6, 6))
plt.plot(history['loss'])
# plt.plot(history['val_loss'])

filename = "Residual1.txt"
history1 = {}
with open(filename, "r") as f:
    history1 = ast.literal_eval(f.read())
plt.plot(history1['loss'])
# plt.plot(history1['val_loss'])



filename = "Residual2.txt"
history2 = {}
with open(filename, "r") as f:
    history2 = ast.literal_eval(f.read())
plt.plot(history2['loss'])
# plt.plot(history2['val_loss'])


filename = "Residual3.txt"
history3 = {}
with open(filename, "r") as f:
    history3 = ast.literal_eval(f.read())
plt.plot(history3['loss'])
# plt.plot(history3['val_loss'])


#
# filename = "Residual4.txt"
# history4 = {}
# with open(filename, "r") as f:
#     history4 = ast.literal_eval(f.read())
# plt.plot(history4['loss'])
# # plt.plot(history3['val_loss'])
#
#
#
#
# filename = "Residual5.txt"
# history5 = {}
# with open(filename, "r") as f:
#     history5 = ast.literal_eval(f.read())
# plt.plot(history5['loss'])
# # plt.plot(history3['val_loss'])

# 设置刻度字体大小

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.axis([0, 80, 0, 11])
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
# plt.title('Skip Transpose IRCNN')
plt.legend([u'lr=0.1', u'lr=0.01', u'lr=0.001', u'lr=0.0001'], fontsize=12)

#plt.legend(['5x5filter_loss', '5x5filter_val_loss',
#            '7x7filter_loss', '7x7filter_val_loss', '9x9filter_loss', '9x9filter_val_loss'])
fig.show()
plt.show()


