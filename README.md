# 资产定价与机器学习：从理论到 A 股实证

自包含教材，配套模拟市场生成器 `apsim`。

- 在线阅读：<https://albertandking.github.io/asset-pricing-ml-book/>（推送到 `main` 后由 GitHub Actions 自动冷构建并发布，见 `.github/workflows/deploy.yml`）
- 许可：正文 CC BY-NC-SA 4.0（见 `LICENSE-CONTENT.md`），代码 MIT（见 `LICENSE`）
- 反馈与勘误：发现错误或有建议，请到 GitHub Issues 提交：<https://github.com/albertandking/asset-pricing-ml-book/issues>

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

`build.sh` 构建后会运行 `scripts/postprocess.sh`：把页面语言改为 `zh-CN`；发布时（见工作流）还会把 sitemap 中 mystmd 写死的 `localhost:3000` 换成线上地址。

构建会缓存代码单元的输出，只有代码单元本身改动时才重新执行。修改了 `src/apsim` 的源码、而章节代码没变时，缓存的输出会过期，这时用 `--clean`。验收必须用 `sh scripts/build.sh --clean`。`--clean` 不会删除 `book/_build/templates`（主题模板缓存），请勿手动删除整个 `book/_build`。

写作时可用实时预览：`cd book && uv run jupyter book start --execute`。

## 测试

```bash
uv run pytest
```

## 目录

- `book/`：书的源文件（MyST Markdown）；目录在 `book/myst.yml` 中
  - `index.md`（前言）、`notation.md`（记号表）、`glossary.md`（术语对照：学术与业界）、`references.bib`（参考文献）
  - `part0/`：第零部分　准备
    - 第 0 章：`ch00-guide.md` 为首页（含本章阅读路径），子页面依次为 `ch00-concepts.md`（0.2 基本概念速览）、`ch00-concepts-sdf.md`（0.2 续：特征、定价误差与随机贴现因子）、`ch00-relations.md`（因子、特征、载荷的关系）、`ch00-model.md`（0.3～0.4 记号与模拟市场的模型）、`ch00-pricing.md`（0.5～0.6 SDF 的定价性质与代码验证）、`ch00-experiments.md`（A 股视角与模拟实验）、`ch00-pitfalls.md`（常见误区）、`ch00-exercises.md`（本章小结、习题与延伸阅读）
    - `primer-regressions.md`：导读：几种回归（第二部分的预习，不属于第 0 章）
  - `part1/`：第一部分　资产定价理论
    - 第 1 章：`ch01-euler-sdf.md` 为首页（学习目标、1.1～1.3 节与本章阅读路径），子页面依次为 `ch01-lognormal.md`（1.4 幂效用与对数正态消费）、`ch01-lucas.md`（1.5～1.6 Lucas 树与回到第 0 章）、`ch01-experiments.md`（A 股视角、模拟实验与常见误区）、`ch01-exercises.md`（本章小结、习题与延伸阅读）
- `src/apsim/`：模拟市场生成器与工具函数（`market`、`consumption`、`stats`、`plotting`）
- `tests/`：单元测试
- `scripts/`：构建、预览与前向引用台账脚本（`build.sh`、`postprocess.sh`、`serve.sh`、`promises.py`）
- `docs/specs/`：设计文档
- `docs/promises.md`：前向引用台账，列出正文中指向尚未写出的章节或附录的引用；由 `uv run python scripts/promises.py` 生成，写每一章之前先运行并核对
