# 电商系统 3 · Boss Agent 研究工作台

Boss Agent 帮助电商团队从公开资料或自行提供的文件形成研究证据、选品机会和报告。工作台保留任务线程、证据来源和执行记录，可按具体模型与工具配置安排数字员工协作。设计任务可以引用研究结果，生成的文件和报告保留在本次安装的工作区。

## 在 Windows 上运行

准备 Python 3.12 以及 Chrome 或 Edge。当前浏览器采集模块使用已安装的浏览器，请在服务所在电脑完成安装。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m app.admin_auth
.\start.ps1
```

终端提示你为 `ADMIN` 设置自己的密码。浏览器访问 `http://127.0.0.1:8765` 后，使用 `ADMIN` 和刚设置的密码登录。工作台只有这一位应用用户；数据库初次启动时建立的「总控 Agent」是任务执行角色。再次运行 `python -m app.admin_auth` 可以更换本机管理员密码。`start.ps1 -Port 8766` 可改用其他端口。

默认 `data/` 保存 SQLite 数据库、本项目自行录入的模型密钥文件 `model-connections.env` 和管理员密码校验值，可以在设置密码与启动前通过 `BOSS_AGENT_DATA` 改用私有目录。任务文件保存在 `workspace/`，设计与导出文件保存在 `outputs/`。`config/settings.yaml` 控制工作预算、浏览器限额和任务调度。干净配置默认暂停自动任务执行、心跳和每日记忆整理，使用者可根据自己的模型与预算配置启用。

## 从证据到报告

1. 首次启动时模型、主路由、备用列表和第三方网页阅读地址均为空。管理员可以在模型与路由设置中主动添加自己的连接并指定任务方案。使用官方客户端时，请在本项目的连接页面完成授权；此部署将官方登录保存在仓库根目录下的 `data/official-client/`。
2. 建立研究项目、指定问题或添加来源，检查提取的证据及适用范围。浏览器模块通过独立浏览器会话读取公开页面。
3. 在工作线程中审阅结果、查看执行记录，生成报告或把机会交给设计任务。`config/design_customer_knowledge.json` 提供空对象示例。有行业知识时，可在本机建立 `config/design_customer_knowledge.local.json`，程序优先读取它，Git 会忽略这个私有文件。

## 参与完善

欢迎在 Issue 分享公开数据源的解析差异、证据去重问题、任务恢复案例或界面改进。一个适中的首个 PR 可以为公开网页解析加入合成 HTML 测试、改善表格读取的边界提示，或补充复核报告中来源的展示。提交时请说明涉及的模块和验证结果，使用虚构的店铺、商品、报告和 API 响应。核心实现和页面模板在 `app/`，现有回归用例在 `tests/`。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests/test_design_service.py
```
