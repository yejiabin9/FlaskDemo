本项目是基于flask搭建的web app
由于之前弄深度学习，生成模型后想要给别人一可视化的方式演示模型的效果
因此基于Python搭建了一个webAPP
特此和大家分享

### 问题一
如果搭建在服务器上面
客户端上传图片的思路是
1、图片下载到服务端
2、服务端加载客户端的图片
这里重复操作了，
服务端直接加载本地的图片就可以了
