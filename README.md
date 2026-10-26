# 资产定价与机器学习：从理论到 A 股实证

本地阅读的自包含教材，配套模拟市场生成器 `apsim`。

## 前提条件

- **uv**：管理 Python 版本与全部依赖（Python 3.13 由 uv 自动安装）。
- **Node.js ≥ 18**：Jupyter Book 2 的构建工具链运行在 Node.js 上（本机为 v24）。
- **首次构建需联网**：构建时会从 GitHub 下载网页主题模板 book-theme，并用 npm 安装其依赖。`book/myst.yml` 中的 `site.template` 把模板固定为 @myst-theme/book 1.4.1（myst-templates/book-theme 仓库的提交 `23f2df5`），模板缓存在 `book/_build/templates/` 下，之后的构建不再联网。若自动下载失败，可手动下载
  `https://codeload.github.com/myst-templates/book-theme/zip/23f2df5493e23dfb1d9433c8aa64c87beb4232a3`，
  解压后把其中 `book-theme-23f2df5…` 目录里的文件（`template.yml`、`server.js` 等）直接放入
  `book/_build/templates/site/3a97af73a992a9550676aa294cc6888a68704eee89dac12946e60abdab0359fd/`
  （目录名是模板 URL 的 SHA-256，由 mystmd 决定），再重新构建。若改回不固定版本的 `template: book-theme`，对应目录为 `book/_build/templates/site/myst/book-theme/`。
- **Windows**：在 Git Bash 中运行 `sh scripts/*.sh`；或直接运行等价命令
  `cd book && uv run jupyter book build --html --execute --strict`。

## 安装

```bash
uv sync
```

## 构建与阅读

```bash
sh scripts/build.sh           # 构建全书（执行全部代码，--strict：任何错误或警告都使构建失败）
sh scripts/build.sh --clean   # 冷构建：先删除执行缓存 book/_build/execute，再构建
sh scripts/serve.sh           # 启动本地服务器，浏览器打开 http://localhost:8000
```

构建会缓存代码单元的输出，只有代码单元本身改动时才重新执行。修改了 `src/apsim` 的源码、而章节代码没变时，缓存的输出会过期，这时用 `--clean`。验收必须用 `sh scripts/build.sh --clean`。`--clean` 不会删除 `book/_build/templates`（主题模板缓存），请勿手动删除整个 `book/_build`。

写作时可用实时预览：`cd book && uv run jupyter book start --execute`。

## 测试

```bash
uv run pytest
```

## 目录

- `book/`：书的源文件（MyST Markdown）
- `src/apsim/`：模拟市场生成器与工具函数
- `tests/`：单元测试
- `docs/specs/`：设计文档
- `参考资料/`：参考材料
