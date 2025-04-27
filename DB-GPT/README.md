环境安装
默认数据库使用SQLite，因此默认启动模式下，无需安装数据库。 我们推荐通过conda的虚拟环境来进行Python虚拟环境的安装。
创建Python虚拟环境
python >= 3.10
conda create -n text_to_sql python=3.10
conda activate text_to_sql

pip install -e ".[default]"


下载Embedding 模型
cd DB-GPT
mkdir models and cd models

#### embedding model
git clone https://huggingface.co/GanymedeNil/text2vec-large-chinese
or
git clone https://huggingface.co/moka-ai/m3e-large


执行脚本初始化表结构, 如果是做版本升级需要提对应的DDL变更来更新表结构
$ mysql -h127.0.0.1 -uroot -p{your_password} < ./assets/schema/dbgpt.sql

如果使用MySQL数据库查询，需要配置MySQL数据库
修改 .env文件 配置MySQL数据库
LOCAL_DB_TYPE=mysql
LOCAL_DB_USER= {your username}
LOCAL_DB_PASSWORD={your_password}
LOCAL_DB_HOST=127.0.0.1
LOCAL_DB_PORT=3306


text_to_sql服务被打包到一个server当中，可以通过如下命令启动text_to_sql系统。
$ python dbgpt/app/dbgpt_server.py

在本地浏览器上打开http://localhost:5670/即可体验text_to_sql系统。