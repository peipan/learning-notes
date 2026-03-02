import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Pan老师的学习笔记",
  description: "Agentic RL + 大模型推理优化",
  base: '/learning-notes/',
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '推理优化', link: '/inference/' },
      { text: 'Agentic RL', link: '/agentic-rl/' },
    ],
    sidebar: {
      '/inference/': [
        {
          text: '推理优化',
          items: [
            { text: '学习路线图', link: '/inference/roadmap' },
            { text: '为什么还需要推理优化', link: '/inference/why-optimize' },
            { text: 'vLLM 部署+源码导读', link: '/inference/vllm-guide' },
            { text: 'SGLang 部署+源码导读', link: '/inference/sglang-guide' },
            { text: '推理优化工程师全景', link: '/inference/engineer-overview' },
            { text: 'Triton 入门教程', link: '/inference/triton-tutorial' },
          ]
        }
      ],
      '/agentic-rl/': [
        {
          text: 'Agentic RL',
          items: [
            { text: '即将更新...', link: '/agentic-rl/' },
          ]
        }
      ],
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/peipan/learning-notes' }
    ],
    outline: {
      level: [2, 3],
      label: '目录'
    },
    search: {
      provider: 'local'
    },
    footer: {
      message: '🐼 Powered by Pander',
    }
  }
})
