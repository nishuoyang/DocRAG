import { chromium } from 'playwright-core'

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
const MARKDOWN = [
  '## 对比结果',
  '',
  '| 城市 | 首付比例 | 说明 |',
  '| --- | --- | --- |',
  '| 北京 | 35% | 普通住宅 |',
  '| 上海 | 30% | 普通住宅 |',
  '',
  '- 数据来自文档',
  '',
  '```text',
  'source: policy.md',
  '```',
  '',
  '<script>window.__markdownXss = true</script>',
].join('\n')

function sse(payload) {
  return `data: ${JSON.stringify(payload)}\n\n`
}

async function mockBackend(page) {
  await page.route('**/chat/memory', (route) =>
    route.fulfill({ json: { messages: [] } })
  )
  await page.route('**/chat/stream', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        sse({ delta: MARKDOWN }),
        sse({ sources: [] }),
        'data: [DONE]\n\n',
      ].join(''),
    })
  )
  await page.route('**/agent/chat/stream', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        sse({ event: 'activity', agent: 'documents', message: '完成' }),
        sse({ event: 'delta', text: MARKDOWN }),
        sse({ event: 'done' }),
      ].join(''),
    })
  )
}

async function submitQuestion(page) {
  const input = page.locator('textarea').first()
  await input.fill('比较北京和上海的首付比例')
  await page.getByRole('button', { name: '发送' }).click()
  await page.waitForTimeout(1500)
}

async function assertMarkdownRendered(page, panelName) {
  const table = page.locator('table').last()
  await table.waitFor({ state: 'visible', timeout: 5000 })
  const headers = await table.locator('thead th').allTextContents()
  const firstRow = await table.locator('tbody tr').first().locator('td').allTextContents()

  if (headers.join('|') !== '城市|首付比例|说明') {
    throw new Error(`${panelName}: 表格表头未正确渲染: ${headers.join('|')}`)
  }
  if (firstRow.join('|') !== '北京|35%|普通住宅') {
    throw new Error(`${panelName}: 表格单元格未正确渲染: ${firstRow.join('|')}`)
  }
  if ((await page.locator('pre code').count()) === 0) {
    throw new Error(`${panelName}: 代码块未正确渲染`)
  }
  if (await page.evaluate(() => window.__markdownXss === true)) {
    throw new Error(`${panelName}: 未过滤 Markdown 中的脚本内容`)
  }
  if ((await page.locator('.markdown-body script').count()) !== 0) {
    throw new Error(`${panelName}: Markdown 输出中仍包含 script 标签`)
  }
}

const browser = await chromium.launch({
  executablePath: CHROME_PATH,
  headless: true,
})
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
await mockBackend(page)
await page.goto('http://localhost:5173')
await page.waitForLoadState('networkidle')

await submitQuestion(page)
await assertMarkdownRendered(page, 'AgentPanel')
await page.screenshot({ path: '../reports/markdown-agent.png', fullPage: true })

await page.getByRole('button', { name: '智能问答' }).click()
await submitQuestion(page)
await assertMarkdownRendered(page, 'ChatPanel')
await page.screenshot({ path: '../reports/markdown-chat.png', fullPage: true })

await browser.close()
console.log('Markdown tables and code blocks rendered correctly in both panels.')
