<div align="center">
  <picture>
      <img src="assets/kimi-logo.png" width="30%" alt="Kimi K3">
  </picture>
</div>
<hr>
<div align="center" style="line-height:1">
  <a href="https://www.kimi.com" target="_blank"><img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-Kimi%20K3-ff6b6b?color=1783ff&logoColor=white"/></a>
  <a href="https://www.moonshot.ai" target="_blank"><img alt="Homepage" src="https://img.shields.io/badge/Homepage-Moonshot%20AI-white?logo=Kimi&logoColor=white"/></a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/moonshotai" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Moonshot%20AI-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://twitter.com/kimi_moonshot" target="_blank"><img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-Kimi.ai-white?logo=x&logoColor=white"/></a>
  <a href="https://discord.gg/TYU2fdJykW" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-Kimi.ai-white?logo=discord&logoColor=white"/></a>
  <a href="https://modelscope.cn/organization/moonshotai" target="_blank"><img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-Moonshot%20AI-white?labelColor=rgb(99%2C%2074%2C%255)"/></a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Kimi_K3-f5de53?&color=f5de53"/></a>
</div>

<p align="center">
  <a href="README.md">English</a> | <b>中文</b>
</p>

<p align="center">
📰&nbsp;&nbsp;<a href="https://www.kimi.com/blog/kimi-k3">技术博客</a> | &nbsp;&nbsp;&nbsp; <b>📄&nbsp;&nbsp;<a href="k3_tech_report.pdf">完整技术报告</a></b>
</p>


## 1. 模型介绍

Kimi K3 是一个开放权重的原生多模态智能体（Agentic）模型，也是我们迄今为止能力最强的模型。它是一个基于 Kimi 增量注意力（Kimi Delta Attention，KDA）和注意力残差（Attention Residuals，AttnRes）构建的 2.8T 参数模型，具备原生视觉能力和 100 万词元（token）的上下文窗口。它是全球首个开放的 3T 级别模型，旨在为长程编码、知识工作和推理等领域提供前沿智能。

### 核心特性
- **全新架构**：Kimi K3 基于 Kimi 增量注意力（KDA）和注意力残差（AttnRes）构建，并通过稳定潜在混合专家（Stable LatentMoE）框架提升混合专家模型（MoE）的稀疏度，在 896 个专家中激活 16 个——相较 Kimi K2，整体扩展效率提升约 2.5 倍。
- **长程编码**：在极少人工干预的情况下，Kimi K3 能够持续进行长时间的工程会话、驾驭超大规模代码仓库并编排终端工具——涵盖图形处理器（GPU）内核优化、编译器开发，乃至视觉参与的游戏开发、计算机辅助设计（CAD）甚至芯片设计。
- **智能体知识工作**：Kimi K3 推动端到端知识工作的发展，依托其原生多模态架构，能够产出带有交互式可视化、组件和仪表盘的深度研究成果，以及动效设计与视频剪辑。
- **原生多模态与长上下文**：Kimi K3 在同一个模型内理解文本、图像和视频，并支持 100 万词元的上下文窗口。
- **开放前沿权重**：我们依据 Kimi K3 许可证发布 Kimi K3 的完整模型权重，让前沿智能开放地服务于研究、部署和进一步创新。
## 2. 模型概要

<div align="center">
<table>
<tbody>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>架构</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">混合专家模型（MoE）</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>总参数量</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">2.8T</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>激活参数量</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">104B</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>层数</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">93</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>稠密层数</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>注意力层构成</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">69 层增量注意力（KDA）+ 24 层门控多头潜在注意力（Gated MLA）</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>注意力隐藏维度</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">7168</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>注意力头数</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">96</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>潜在混合专家（Latent MoE）维度</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">3584</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>混合专家（MoE）隐藏维度</strong>（每个专家）</td>
<td align="center" style="vertical-align: middle; text-align: center">3072</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>专家数量</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">896</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>每个词元选择的专家数</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">16</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>共享专家数量</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>词表大小</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">160K</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>上下文长度</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">1048576</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>注意力机制</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">增量注意力（KDA）&amp; 门控多头潜在注意力（Gated MLA）</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>激活函数</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">SiTU-GLU</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>视觉编码器</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">MoonViT-V2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>视觉编码器参数量</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">401M</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>量化</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">四位微缩浮点（MXFP4）权重 / 八位微缩浮点（MXFP8）激活<br>（量化感知训练）</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>模态</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">文本、图像</td>
</tr>
</tbody>
</table>
</div>


## 3. 评测结果

<div align="center">
<table>
<thead>
<tr>
<th align="center" style="text-align: center">基准测试</th>
<th align="center" style="text-align: center"><sup>Kimi K3<br><sup>(max)</sup></sup></th>
<th align="center" style="text-align: center"><sup>Claude Fable 5<br><sup>(max, w/ fallback)</sup></sup></th>
<th align="center" style="text-align: center"><sup>GPT-5.6 Sol<br><sup>(max)</sup></sup></th>
<th align="center" style="text-align: center"><sup>Claude Opus 4.8<br><sup>(max)</sup></sup></th>
<th align="center" style="text-align: center"><sup>GPT-5.5<br><sup>(xhigh)</sup></sup></th>
<th align="center" style="text-align: center"><sup>GLM-5.2<br><sup>(max)</sup></sup></th>
</tr>
</thead>
<tbody>
<tr>
<td align="center" colspan=7 style="text-align: center"><strong>推理与知识</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">GPQA Diamond</td>
<td align="center" style="vertical-align: middle; text-align: center">93.5</td>
<td align="center" style="vertical-align: middle; text-align: center">92.6</td>
<td align="center" style="vertical-align: middle; text-align: center">94.1</td>
<td align="center" style="vertical-align: middle; text-align: center">91.0</td>
<td align="center" style="vertical-align: middle; text-align: center">93.5</td>
<td align="center" style="vertical-align: middle; text-align: center">91.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">CritPt</td>
<td align="center" style="vertical-align: middle; text-align: center">23.4</td>
<td align="center" style="vertical-align: middle; text-align: center">28.6</td>
<td align="center" style="vertical-align: middle; text-align: center">32.3</td>
<td align="center" style="vertical-align: middle; text-align: center">20.9</td>
<td align="center" style="vertical-align: middle; text-align: center">27.1</td>
<td align="center" style="vertical-align: middle; text-align: center">20.9</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">AA-LCR</td>
<td align="center" style="vertical-align: middle; text-align: center">74.7</td>
<td align="center" style="vertical-align: middle; text-align: center">70.0</td>
<td align="center" style="vertical-align: middle; text-align: center">73.7</td>
<td align="center" style="vertical-align: middle; text-align: center">67.7</td>
<td align="center" style="vertical-align: middle; text-align: center">74.3</td>
<td align="center" style="vertical-align: middle; text-align: center">71.3</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">HLE-Full</td>
<td align="center" style="vertical-align: middle; text-align: center">43.5 / 56.0</td>
<td align="center" style="vertical-align: middle; text-align: center">53.3 / 63.0</td>
<td align="center" style="vertical-align: middle; text-align: center">44.5 / 58.0</td>
<td align="center" style="vertical-align: middle; text-align: center">49.8 / 57.9</td>
<td align="center" style="vertical-align: middle; text-align: center">41.4 / 52.2</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" colspan=7 style="text-align: center"><strong>编码</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">DeepSWE</td>
<td align="center" style="vertical-align: middle; text-align: center">67.5</td>
<td align="center" style="vertical-align: middle; text-align: center">70.0</td>
<td align="center" style="vertical-align: middle; text-align: center">73.0</td>
<td align="center" style="vertical-align: middle; text-align: center">59.0</td>
<td align="center" style="vertical-align: middle; text-align: center">67.0</td>
<td align="center" style="vertical-align: middle; text-align: center">46.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">ProgramBench</td>
<td align="center" style="vertical-align: middle; text-align: center">77.8</td>
<td align="center" style="vertical-align: middle; text-align: center">76.8</td>
<td align="center" style="vertical-align: middle; text-align: center">77.6</td>
<td align="center" style="vertical-align: middle; text-align: center">71.9</td>
<td align="center" style="vertical-align: middle; text-align: center">70.8</td>
<td align="center" style="vertical-align: middle; text-align: center">63.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Terminal-Bench 2.1</td>
<td align="center" style="vertical-align: middle; text-align: center">88.3</td>
<td align="center" style="vertical-align: middle; text-align: center">88.0</td>
<td align="center" style="vertical-align: middle; text-align: center">88.8</td>
<td align="center" style="vertical-align: middle; text-align: center">84.6</td>
<td align="center" style="vertical-align: middle; text-align: center">83.4</td>
<td align="center" style="vertical-align: middle; text-align: center">82.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">FrontierSWE</td>
<td align="center" style="vertical-align: middle; text-align: center">81.2</td>
<td align="center" style="vertical-align: middle; text-align: center">86.6</td>
<td align="center" style="vertical-align: middle; text-align: center">71.3</td>
<td align="center" style="vertical-align: middle; text-align: center">66.7</td>
<td align="center" style="vertical-align: middle; text-align: center">64.9</td>
<td align="center" style="vertical-align: middle; text-align: center">67.3</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">SWE-Marathon</td>
<td align="center" style="vertical-align: middle; text-align: center">42.0</td>
<td align="center" style="vertical-align: middle; text-align: center">35.0</td>
<td align="center" style="vertical-align: middle; text-align: center">39.0</td>
<td align="center" style="vertical-align: middle; text-align: center">40.0</td>
<td align="center" style="vertical-align: middle; text-align: center">14.0</td>
<td align="center" style="vertical-align: middle; text-align: center">13.0</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">PostTrainBench</td>
<td align="center" style="vertical-align: middle; text-align: center">36.6</td>
<td align="center" style="vertical-align: middle; text-align: center">41.4</td>
<td align="center" style="vertical-align: middle; text-align: center">34.6</td>
<td align="center" style="vertical-align: middle; text-align: center">34.1</td>
<td align="center" style="vertical-align: middle; text-align: center">28.4</td>
<td align="center" style="vertical-align: middle; text-align: center">34.3</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MLS-Bench-Lite</td>
<td align="center" style="vertical-align: middle; text-align: center">48.3</td>
<td align="center" style="vertical-align: middle; text-align: center">49.9</td>
<td align="center" style="vertical-align: middle; text-align: center">46.2</td>
<td align="center" style="vertical-align: middle; text-align: center">42.8</td>
<td align="center" style="vertical-align: middle; text-align: center">35.5</td>
<td align="center" style="vertical-align: middle; text-align: center">40.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">SciCode</td>
<td align="center" style="vertical-align: middle; text-align: center">58.7</td>
<td align="center" style="vertical-align: middle; text-align: center">60.2</td>
<td align="center" style="vertical-align: middle; text-align: center">56.1</td>
<td align="center" style="vertical-align: middle; text-align: center">53.5</td>
<td align="center" style="vertical-align: middle; text-align: center">56.1</td>
<td align="center" style="vertical-align: middle; text-align: center">50.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Kimi Code Bench 2.0</td>
<td align="center" style="vertical-align: middle; text-align: center">72.9</td>
<td align="center" style="vertical-align: middle; text-align: center">76.9</td>
<td align="center" style="vertical-align: middle; text-align: center">64.8</td>
<td align="center" style="vertical-align: middle; text-align: center">71.7</td>
<td align="center" style="vertical-align: middle; text-align: center">69.0</td>
<td align="center" style="vertical-align: middle; text-align: center">64.2</td>
</tr>
<tr>
<td align="center" colspan=7 style="text-align: center"><strong>智能体</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">BrowseComp</td>
<td align="center" style="vertical-align: middle; text-align: center">91.2</td>
<td align="center" style="vertical-align: middle; text-align: center">88.0</td>
<td align="center" style="vertical-align: middle; text-align: center">90.4</td>
<td align="center" style="vertical-align: middle; text-align: center">84.3</td>
<td align="center" style="vertical-align: middle; text-align: center">84.4</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">DeepSearchQA (F1)</td>
<td align="center" style="vertical-align: middle; text-align: center">95.0</td>
<td align="center" style="vertical-align: middle; text-align: center">94.2</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">93.1</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">ResearchRubrics</td>
<td align="center" style="vertical-align: middle; text-align: center">76.2</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">73.8</td>
<td align="center" style="vertical-align: middle; text-align: center">73.5</td>
<td align="center" style="vertical-align: middle; text-align: center">64.0</td>
<td align="center" style="vertical-align: middle; text-align: center">71.1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">GDPval-AA v2 (Elo)</td>
<td align="center" style="vertical-align: middle; text-align: center">1686</td>
<td align="center" style="vertical-align: middle; text-align: center">1747</td>
<td align="center" style="vertical-align: middle; text-align: center">1736</td>
<td align="center" style="vertical-align: middle; text-align: center">1593</td>
<td align="center" style="vertical-align: middle; text-align: center">1491</td>
<td align="center" style="vertical-align: middle; text-align: center">1510</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Toolathlon-Verified</td>
<td align="center" style="vertical-align: middle; text-align: center">76.5</td>
<td align="center" style="vertical-align: middle; text-align: center">77.9</td>
<td align="center" style="vertical-align: middle; text-align: center">74.9</td>
<td align="center" style="vertical-align: middle; text-align: center">76.2</td>
<td align="center" style="vertical-align: middle; text-align: center">73.5</td>
<td align="center" style="vertical-align: middle; text-align: center">59.9</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MCPMark-Verified</td>
<td align="center" style="vertical-align: middle; text-align: center">94.5</td>
<td align="center" style="vertical-align: middle; text-align: center">87.4</td>
<td align="center" style="vertical-align: middle; text-align: center">92.9</td>
<td align="center" style="vertical-align: middle; text-align: center">76.4</td>
<td align="center" style="vertical-align: middle; text-align: center">92.9</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MCP-Atlas</td>
<td align="center" style="vertical-align: middle; text-align: center">84.2</td>
<td align="center" style="vertical-align: middle; text-align: center">84.7</td>
<td align="center" style="vertical-align: middle; text-align: center">83.6</td>
<td align="center" style="vertical-align: middle; text-align: center">83.6</td>
<td align="center" style="vertical-align: middle; text-align: center">82.8</td>
<td align="center" style="vertical-align: middle; text-align: center">82.6</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">AutomationBench</td>
<td align="center" style="vertical-align: middle; text-align: center">30.8</td>
<td align="center" style="vertical-align: middle; text-align: center">29.1</td>
<td align="center" style="vertical-align: middle; text-align: center">29.7</td>
<td align="center" style="vertical-align: middle; text-align: center">27.2</td>
<td align="center" style="vertical-align: middle; text-align: center">22.7</td>
<td align="center" style="vertical-align: middle; text-align: center">12.9</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">JobBench</td>
<td align="center" style="vertical-align: middle; text-align: center">54.3</td>
<td align="center" style="vertical-align: middle; text-align: center">57.4</td>
<td align="center" style="vertical-align: middle; text-align: center">45.4</td>
<td align="center" style="vertical-align: middle; text-align: center">48.4</td>
<td align="center" style="vertical-align: middle; text-align: center">38.3</td>
<td align="center" style="vertical-align: middle; text-align: center">43.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">AA-Briefcase (Elo)</td>
<td align="center" style="vertical-align: middle; text-align: center">1548</td>
<td align="center" style="vertical-align: middle; text-align: center">1583</td>
<td align="center" style="vertical-align: middle; text-align: center">1495</td>
<td align="center" style="vertical-align: middle; text-align: center">1354</td>
<td align="center" style="vertical-align: middle; text-align: center">1158</td>
<td align="center" style="vertical-align: middle; text-align: center">1260</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Agents' Last Exam</td>
<td align="center" style="vertical-align: middle; text-align: center">28.3</td>
<td align="center" style="vertical-align: middle; text-align: center">25.7<sup>†</sup></td>
<td align="center" style="vertical-align: middle; text-align: center">29.6</td>
<td align="center" style="vertical-align: middle; text-align: center">27.0</td>
<td align="center" style="vertical-align: middle; text-align: center">26.6</td>
<td align="center" style="vertical-align: middle; text-align: center">20.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">APEX-Agents</td>
<td align="center" style="vertical-align: middle; text-align: center">41.0</td>
<td align="center" style="vertical-align: middle; text-align: center">43.3</td>
<td align="center" style="vertical-align: middle; text-align: center">39.9</td>
<td align="center" style="vertical-align: middle; text-align: center">39.4</td>
<td align="center" style="vertical-align: middle; text-align: center">38.5</td>
<td align="center" style="vertical-align: middle; text-align: center">35.6</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">OfficeQA Pro</td>
<td align="center" style="vertical-align: middle; text-align: center">63.3</td>
<td align="center" style="vertical-align: middle; text-align: center">69.9</td>
<td align="center" style="vertical-align: middle; text-align: center">63.2</td>
<td align="center" style="vertical-align: middle; text-align: center">63.9</td>
<td align="center" style="vertical-align: middle; text-align: center">60.9</td>
<td align="center" style="vertical-align: middle; text-align: center">41.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">SpreadsheetBench 2</td>
<td align="center" style="vertical-align: middle; text-align: center">34.8</td>
<td align="center" style="vertical-align: middle; text-align: center">34.7</td>
<td align="center" style="vertical-align: middle; text-align: center">32.4</td>
<td align="center" style="vertical-align: middle; text-align: center">31.6</td>
<td align="center" style="vertical-align: middle; text-align: center">29.1</td>
<td align="center" style="vertical-align: middle; text-align: center">28.1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">OSWorld-Verified</td>
<td align="center" style="vertical-align: middle; text-align: center">84.8</td>
<td align="center" style="vertical-align: middle; text-align: center">85.0</td>
<td align="center" style="vertical-align: middle; text-align: center">83.0</td>
<td align="center" style="vertical-align: middle; text-align: center">83.4</td>
<td align="center" style="vertical-align: middle; text-align: center">79.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">OSWorld 2.0</td>
<td align="center" style="vertical-align: middle; text-align: center">58.3</td>
<td align="center" style="vertical-align: middle; text-align: center">66.1</td>
<td align="center" style="vertical-align: middle; text-align: center">62.6</td>
<td align="center" style="vertical-align: middle; text-align: center">55.7</td>
<td align="center" style="vertical-align: middle; text-align: center">49.5</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">SaaS-Bench</td>
<td align="center" style="vertical-align: middle; text-align: center">60.1</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">61.4</td>
<td align="center" style="vertical-align: middle; text-align: center">56.1</td>
<td align="center" style="vertical-align: middle; text-align: center">43.8</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">τ³-Banking</td>
<td align="center" style="vertical-align: middle; text-align: center">33.4</td>
<td align="center" style="vertical-align: middle; text-align: center">26.8</td>
<td align="center" style="vertical-align: middle; text-align: center">33.0</td>
<td align="center" style="vertical-align: middle; text-align: center">27.6</td>
<td align="center" style="vertical-align: middle; text-align: center">31.3</td>
<td align="center" style="vertical-align: middle; text-align: center">26.8</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Harvey Lab-AA</td>
<td align="center" style="vertical-align: middle; text-align: center">94.6</td>
<td align="center" style="vertical-align: middle; text-align: center">93.6</td>
<td align="center" style="vertical-align: middle; text-align: center">87.2</td>
<td align="center" style="vertical-align: middle; text-align: center">91.1</td>
<td align="center" style="vertical-align: middle; text-align: center">86.3</td>
<td align="center" style="vertical-align: middle; text-align: center">91.0</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">CorpFin v2</td>
<td align="center" style="vertical-align: middle; text-align: center">71.6</td>
<td align="center" style="vertical-align: middle; text-align: center">71.8</td>
<td align="center" style="vertical-align: middle; text-align: center">64.4</td>
<td align="center" style="vertical-align: middle; text-align: center">66.7</td>
<td align="center" style="vertical-align: middle; text-align: center">68.4</td>
<td align="center" style="vertical-align: middle; text-align: center">66.1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Finance Agent v2</td>
<td align="center" style="vertical-align: middle; text-align: center">54.4</td>
<td align="center" style="vertical-align: middle; text-align: center">56.3</td>
<td align="center" style="vertical-align: middle; text-align: center">53.8</td>
<td align="center" style="vertical-align: middle; text-align: center">53.9</td>
<td align="center" style="vertical-align: middle; text-align: center">51.8</td>
<td align="center" style="vertical-align: middle; text-align: center">49.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Legal Research Bench</td>
<td align="center" style="vertical-align: middle; text-align: center">44.2</td>
<td align="center" style="vertical-align: middle; text-align: center">49.5</td>
<td align="center" style="vertical-align: middle; text-align: center">48.1</td>
<td align="center" style="vertical-align: middle; text-align: center">43.8</td>
<td align="center" style="vertical-align: middle; text-align: center">40.4</td>
<td align="center" style="vertical-align: middle; text-align: center">31.3</td>
</tr>
<tr>
<td align="center" colspan=7 style="text-align: center"><strong>视觉</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">WorldVQA ForceAnswer</td>
<td align="center" style="vertical-align: middle; text-align: center">51.0</td>
<td align="center" style="vertical-align: middle; text-align: center">56.7</td>
<td align="center" style="vertical-align: middle; text-align: center">41.8</td>
<td align="center" style="vertical-align: middle; text-align: center">39.1</td>
<td align="center" style="vertical-align: middle; text-align: center">38.5</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">OmniDocBench</td>
<td align="center" style="vertical-align: middle; text-align: center">91.1</td>
<td align="center" style="vertical-align: middle; text-align: center">89.8</td>
<td align="center" style="vertical-align: middle; text-align: center">85.8</td>
<td align="center" style="vertical-align: middle; text-align: center">87.9</td>
<td align="center" style="vertical-align: middle; text-align: center">89.4</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">PerceptionBench</td>
<td align="center" style="vertical-align: middle; text-align: center">58.5</td>
<td align="center" style="vertical-align: middle; text-align: center">57.2</td>
<td align="center" style="vertical-align: middle; text-align: center">59.7</td>
<td align="center" style="vertical-align: middle; text-align: center">47.2</td>
<td align="center" style="vertical-align: middle; text-align: center">55.8</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Video-MME (w. sub)</td>
<td align="center" style="vertical-align: middle; text-align: center">90.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">89.5</td>
<td align="center" style="vertical-align: middle; text-align: center">86.0</td>
<td align="center" style="vertical-align: middle; text-align: center">89.3</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MMVU</td>
<td align="center" style="vertical-align: middle; text-align: center">82.1</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">81.2</td>
<td align="center" style="vertical-align: middle; text-align: center">79.2</td>
<td align="center" style="vertical-align: middle; text-align: center">81.7</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">BabyVision w/ python</td>
<td align="center" style="vertical-align: middle; text-align: center">85.7</td>
<td align="center" style="vertical-align: middle; text-align: center">90.5</td>
<td align="center" style="vertical-align: middle; text-align: center">88.9</td>
<td align="center" style="vertical-align: middle; text-align: center">81.2</td>
<td align="center" style="vertical-align: middle; text-align: center">83.6</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MMMU-Pro</td>
<td align="center" style="vertical-align: middle; text-align: center">81.6 / 83.4</td>
<td align="center" style="vertical-align: middle; text-align: center">81.2 / 86.5</td>
<td align="center" style="vertical-align: middle; text-align: center">83.0 / 84.6</td>
<td align="center" style="vertical-align: middle; text-align: center">78.9 / 82.7</td>
<td align="center" style="vertical-align: middle; text-align: center">81.2 / 83.2</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">CharXiv (RQ)</td>
<td align="center" style="vertical-align: middle; text-align: center">84.8 / 91.3</td>
<td align="center" style="vertical-align: middle; text-align: center">88.9 / 93.5</td>
<td align="center" style="vertical-align: middle; text-align: center">84.6 / 89.1</td>
<td align="center" style="vertical-align: middle; text-align: center">80.5 / 89.9</td>
<td align="center" style="vertical-align: middle; text-align: center">84.1 / 89.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MathVision</td>
<td align="center" style="vertical-align: middle; text-align: center">94.3 / 97.8</td>
<td align="center" style="vertical-align: middle; text-align: center">94.8 / 98.6</td>
<td align="center" style="vertical-align: middle; text-align: center">95.8 / 97.8</td>
<td align="center" style="vertical-align: middle; text-align: center">86.7 / 97.1</td>
<td align="center" style="vertical-align: middle; text-align: center">92.2 / 96.8</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">ZeroBench (pass@5)</td>
<td align="center" style="vertical-align: middle; text-align: center">23.0 / 41.0</td>
<td align="center" style="vertical-align: middle; text-align: center">23.0 / 46.0</td>
<td align="center" style="vertical-align: middle; text-align: center">17.0 / 35.0</td>
<td align="center" style="vertical-align: middle; text-align: center">17.0 / 34.0</td>
<td align="center" style="vertical-align: middle; text-align: center">22.0 / 41.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
</tbody>
</table>
</div>

<details>
<summary><b>脚注</b></summary>

Kimi K3 的所有结果均在推理强度（reasoning effort）设为 'max'、温度（temperature）= 1.0 的条件下获得。对于单步任务（如 GPQA Diamond、HLE-Full 以及不使用工具的视觉基准测试），我们设置核采样（top-p）= 0.95；对于智能体任务，我们设置核采样（top-p）= 1.0。对于 HLE-Full、MMMU-Pro、CharXiv (RQ)、MathVision 和 ZeroBench，每个单元格依次报告不使用和使用工具增强（HLE-Full 使用通用工具，视觉基准测试使用 Python）的分数。

1. **推理与知识基准测试**
   - **CritPt 和 AA-LCR。** 分数引自 [Artificial Analysis](https://artificialanalysis.ai/)，截至 2026 年 7 月 23 日。
2. **编码基准测试**
   - **DeepSWE。** Kimi K3 使用 Kimi Code 框架进行评测。GLM-5.2 的分数取自 [GLM-5.2 发布博客](https://z.ai/blog/glm-5.2)；其余所有分数均来自官方 [DeepSWE 排行榜](https://deepswe.datacurve.ai/)，Kimi K3 在该排行榜上使用 mini-SWE-agent 框架取得 67.3 分。我们报告的是 DeepSWE v1.1 任务的结果。
   - **Terminal-Bench 2.1。** Kimi K3 使用 Kimi Code 框架进行评测。对于其他所有模型，我们报告其在各框架下的最佳分数：GLM-5.2 使用 Claude Code（[GLM-5.2 发布博客](https://z.ai/blog/glm-5.2)）；Claude Opus 4.8 和 Claude Fable 5 使用 Terminus 2（[Artificial Analysis](https://artificialanalysis.ai/evaluations/terminalbench-v2-1)）；GPT-5.5 和 GPT-5.6 Sol 使用 Codex（[OpenAI](https://openai.com/index/previewing-gpt-5-6-sol/)）。
   - **ProgramBench。** Kimi K3 使用 Kimi Code 框架进行评测。GLM-5.2 的分数来自 [GLM-5.2 发布博客](https://z.ai/blog/glm-5.2)；其他所有分数来自 [Vals AI](https://www.vals.ai/benchmarks/programbench)。
   - **SWE-Marathon。** Kimi K3、Claude Opus 4.8 和 Claude Fable 5 使用 Claude Code 框架进行评测；GPT-5.6 Sol 使用 Codex 框架进行评测。GLM-5.2 的分数来自 [GLM-5.2 发布博客](https://z.ai/blog/glm-5.2)。我们的评测基于[官方任务](https://www.swe-marathon.org/)在 2026 年 7 月 9 日（最终 v1.1 版本发布之前）的一个针对 H20 校准的分支：GPU 任务的 Docker 镜像、性能门槛和参考基准已针对 H20 重新校准，而正确性和防作弊验证器保持不变。此外，在我们的评测中，Claude Fable 5 在 35% 的任务上触发了回退（fallback），这可能对其测得的性能产生负面影响。
   - **FrontierSWE。** Kimi K3 使用 Kimi Code 框架、GPT-5.6 Sol 使用 Codex 框架进行评测；其他所有结果来自 [FrontierSWE](https://www.frontierswe.com/)。优势分数（Dominance scores）使用官方评测脚本从原始分数重新计算，数据截至 2026 年 7 月 16 日。
   - **PostTrainBench。** GLM-5.2、GPT-5.5 和 Claude Opus 4.8 的分数采用官方 [PostTrainBench](https://posttrainbench.com/) 结果。Kimi K3、Claude Fable 5 和 GPT-5.6 Sol 使用官方 Harbor 实现，在最大推理强度下评测，在 H20 GPU（而非官方设置中的 H100）上运行三次取平均——其中 Kimi K3 和 Claude Fable 5 使用 Claude Code 框架，GPT-5.6 Sol 使用 Codex 框架。
   - **MLS-Bench-Lite。** Kimi K3 使用 Kimi Code 框架进行评测；GLM-5.2 和 Claude 系列模型使用 Claude Code 框架；GPT-5.5 和 GPT-5.6 Sol 使用 Codex 框架。
   - **SciCode。** 分数引自 [Artificial Analysis](https://artificialanalysis.ai/)，截至 2026 年 7 月 23 日。
   - **Kimi Code Bench 2.0（内部基准测试）。** Kimi K3 使用 Kimi Code 框架进行评测（使用 Claude Code 框架时取得 73.7 分）；GLM-5.2、Claude Opus 4.8 和 Claude Fable 5 使用 Claude Code 框架；GPT-5.5 和 GPT-5.6 Sol 使用 Codex 框架。所有模型均在最大推理强度下评测，GPT-5.5 除外（使用 "xhigh" 设置）。由于该基准测试包含网络安全和安全相关任务，我们同时披露被拒绝或触发回退的任务比例：Claude Fable 5 在 80 个任务中触发 13 次回退和 1 次拒绝；GPT-5.6 Sol 的网络安全防护在 80 个任务中触发 10 次拒绝；GPT-5.5 在 80 个任务中出现 3 次拒绝。
3. **智能体基准测试**
   - **OfficeQA Pro。** 每个测试用例向智能体提供完整的 PDF 语料库，所有 PDF 均以图像形式渲染，不提供机器可读文本。
   - **OfficeQA Pro 和 SpreadsheetBench 2。** Kimi K3、GLM-5.2、Claude Opus 4.8 和 Claude Fable 5 使用 Claude Code 框架进行评测；GPT-5.5 和 GPT-5.6 Sol 使用 Codex 框架进行评测。
   - **MCP-Atlas。** 所有模型均在 500 个任务的公开子集上评测，回合上限为 100，使用 Gemini 3.1 Pro 作为评判模型。
   - **AutomationBench。** 所有模型均在 600 个任务的公开子集上评测，其余方面均遵循官方 GitHub 设置。
   - **BrowseComp。** 我们采用在 30 万词元时触发的上下文压缩策略。在使用完整 100 万词元上下文窗口且不进行上下文管理的情况下评测时，Kimi K3 取得 90.4 分。Claude Fable 5、Claude Opus 4.8、GPT-5.6 Sol 和 GPT-5.5 的结果引自 [Anthropic](https://www.anthropic.com/news/claude-fable-5-mythos-5) 和 [OpenAI](https://openai.com/index/gpt-5-6/)。
   - **GDPval-AA v2、AA-Briefcase、τ³-Banking、Harvey Lab-AA 和 APEX-Agents。** 分数引自 [Artificial Analysis](https://artificialanalysis.ai/) 和 [APEX-Agents 排行榜](https://www.mercor.com/apex/apex-agents-leaderboard/)，截至 2026 年 7 月 23 日。对于 Harvey Lab-AA，我们报告标准通过率（criterion pass rate）。
   - **CorpFin v2、Finance Agent v2 和 Legal Research Bench。** 分数引自 [Vals AI](https://www.vals.ai/)。
   - **Agents' Last Exam。** 分数引自[官方排行榜](https://agents-last-exam.org/leaderboard)，截至 2026 年 7 月 23 日；我们报告排行榜的主要通过率指标。在排行榜上，每个模型与特定框架配对：Kimi K3 使用 Kimi Code；GPT-5.6 Sol 和 GPT-5.5 使用 Codex；Claude Fable 5、Claude Opus 4.8 和 GLM-5.2 使用 Claude Code。<sup>†</sup> Claude Fable 5 条目以 xhigh 强度运行，其中 40% 的任务被标注为降级。
4. **多模态基准测试**
   - 除 ZeroBench 遵循官方设置运行五次外，所有多模态分数均为三次运行的平均值。MMMU-Pro 按照官方协议评测，保留原始输入顺序并将图像置于文本输入之前。
   - **PerceptionBench** 是一个专注于原子级视觉感知能力的内部基准测试。

</details>

## 4. 原生四位微缩浮点（MXFP4）量化

Kimi K3 从监督微调（SFT）阶段起就采用量化感知训练，使用四位微缩浮点（MXFP4）权重与八位微缩浮点（MXFP8）激活，以实现广泛的硬件兼容性。

## 5. 部署

> [!Note]
> 您可以在 https://platform.kimi.ai 上选择 `kimi-k3` 来访问 Kimi K3 的应用程序接口（API），我们为您提供 OpenAI/Anthropic 兼容的接口。目前，推荐使用以下推理引擎运行 Kimi K3：

- [vLLM](https://github.com/vllm-project/vllm) — 参见 [recipes](https://recipes.vllm.ai/moonshotai/Kimi-K3)
- [SGLang](https://github.com/sgl-project/sglang) — 参见 [cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)
- [TokenSpeed](https://lightseek.org/tokenspeed) — 参见 [recipes](https://lightseek.org/tokenspeed/recipes/models#kimi-k3)

---
## 6. 模型使用

Kimi K3 始终开启思考（thinking）模式，并会返回 `reasoning_content`。思考强度通过请求中的顶层字段 `reasoning_effort` 配置，支持 `"low"`、`"high"` 和 `"max"`（默认为 `"max"`）。

Kimi K3 以保留思考历史（preserved thinking history）模式训练。在多轮对话和工具调用中，Kimi K3 要求将接口返回的完整助手（assistant）消息原样传回 `messages`——包括思考内容（`reasoning_content`）和工具调用（`tool_calls`），而不仅仅是回复内容（`content`）：

```python
import openai

def chat_with_preserved_thinking(client: openai.OpenAI, model_name: str):
    messages = [
        {
            "role": "user",
            "content": "Tell me three random numbers."
        },
        {
            "role": "assistant",
            "reasoning_content": "I'll start by listing five numbers: 473, 921, 235, 215, 222, and I'll tell you the first three.",
            "content": "473, 921, 235"
        },
        {
            "role": "user",
            "content": "What are the other two numbers you have in mind?"
        }
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        max_tokens=4096,
        reasoning_effort="max",
    )
    # the assistant should mention 215 and 222 that appear in the prior reasoning content
    print(f"response: {response.choices[0].message.reasoning}")
    return response.choices[0].message.content
```

完整的指南和示例（视觉输入、结构化输出、部分补全（partial）模式、工具选择、动态工具加载、上下文缓存）请参见 [Kimi K3 快速入门](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)和[思考强度](https://platform.kimi.ai/docs/guide/use-thinking-effort)。

### 编码智能体框架

Kimi K3 与 [Kimi Code CLI](https://www.kimi.com/code) 智能体框架搭配使用效果最佳。我们诚挚邀请您试用——在终端中运行 Kimi Code，并使用 `/model` 命令选择 Kimi K3。希望您享受与 Kimi K3 一起构建的过程，我们也期待听到您的反馈！


---

## 7. 许可证

代码仓库和模型权重均依据 [Kimi K3 许可证](LICENSE)发布。

---

## 8. 联系我们

如有任何问题，请通过 [support@moonshot.ai](mailto:support@moonshot.ai) 联系我们。
