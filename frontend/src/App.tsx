import { useEffect, useMemo, useState } from 'react'
import {
  App as AntApp, Avatar, Button, Card, Checkbox, Col, Form, Input, Layout, Menu,
  Progress, Radio, Row, Select, Space, Steps, Table, Tabs, Tag, Typography, Upload,
} from 'antd'
import type { TableColumnsType, UploadProps } from 'antd'
import {
  AudioOutlined, BellOutlined, CloudUploadOutlined, DatabaseOutlined, FileTextOutlined,
  FolderOpenOutlined, HomeOutlined, InboxOutlined, PlayCircleFilled, RobotOutlined,
  SaveOutlined, SettingOutlined, StopFilled, TeamOutlined, TranslationOutlined,
} from '@ant-design/icons'

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

type RowData = { key: number; id: number; start: string; end: string; speaker: string; source: string; target: string; status: string }
type Task = { state: string; message: string; backend_status?: string; result?: any }

const languages = ['中文', '英语', '日语', '韩语']
const rowFromApi = (row: any[]): RowData => ({ key: Number(row[0]), id: Number(row[0]), start: row[1], end: row[2], speaker: row[3], source: row[4], target: row[5], status: row[6] })
const rowToApi = (row: RowData) => [row.id, row.start, row.end, row.speaker, row.source, row.target, row.status]

async function jsonFetch(url: string, init?: RequestInit) {
  const response = await fetch(url, { headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) }, ...init })
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `请求失败 ${response.status}`)
  return response.json()
}

function App() {
  const { message } = AntApp.useApp()
  const [nav, setNav] = useState('studio')
  const [stage, setStage] = useState(0)
  const [projectId, setProjectId] = useState('')
  const [mediaPath, setMediaPath] = useState('')
  const [refPath, setRefPath] = useState('')
  const [rows, setRows] = useState<RowData[]>([])
  const [busy, setBusy] = useState(false)
  const [taskText, setTaskText] = useState('等待任务')
  const [progress, setProgress] = useState(0)
  const [history, setHistory] = useState<any[]>([])
  const [models, setModels] = useState<any[]>([])
  const [form] = Form.useForm()

  const updateRow = (key: number, field: keyof RowData, value: string) => {
    setRows(valueRows => valueRows.map(row => row.key === key ? { ...row, [field]: value } : row))
  }
  const columns: TableColumnsType<RowData> = useMemo(() => [
    { title: '#', dataIndex: 'id', width: 54, fixed: 'left' },
    { title: '开始', dataIndex: 'start', width: 105 },
    { title: '结束', dataIndex: 'end', width: 105 },
    { title: '角色', dataIndex: 'speaker', width: 120, render: (_, r) => <Input value={r.speaker} onChange={e => updateRow(r.key, 'speaker', e.target.value)} /> },
    { title: '识别原文', dataIndex: 'source', width: 330, render: (_, r) => <Input.TextArea autoSize={{ minRows: 1, maxRows: 4 }} value={r.source} onChange={e => updateRow(r.key, 'source', e.target.value)} /> },
    { title: '目标字幕', dataIndex: 'target', width: 370, render: (_, r) => <Input.TextArea autoSize={{ minRows: 1, maxRows: 4 }} value={r.target} onChange={e => updateRow(r.key, 'target', e.target.value)} /> },
    { title: '状态', dataIndex: 'status', width: 110, render: value => <Tag color={value.includes('完成') || value.includes('合成') ? 'green' : 'blue'}>{value}</Tag> },
  ], [])

  useEffect(() => {
    jsonFetch('/api/history').then(setHistory).catch(() => {})
    jsonFetch('/api/models').then(setModels).catch(() => {})
  }, [])

  const pollTask = async (taskId: string, onDone: (result: any) => void) => {
    setBusy(true); setProgress(12)
    const timer = window.setInterval(async () => {
      try {
        const task: Task = await jsonFetch(`/api/tasks/${taskId}`)
        setTaskText(task.backend_status || task.message)
        setProgress(value => Math.min(92, value + 4))
        if (task.state === 'completed') {
          window.clearInterval(timer); setBusy(false); setProgress(100); onDone(task.result); message.success(task.result?.message || '任务完成')
        } else if (task.state === 'failed') {
          window.clearInterval(timer); setBusy(false); setProgress(0); message.error(task.message)
        }
      } catch (error: any) { window.clearInterval(timer); setBusy(false); message.error(error.message) }
    }, 1000)
  }

  const uploadProps = (kind: 'media' | 'reference'): UploadProps => ({
    multiple: false, showUploadList: true,
    customRequest: async options => {
      const data = new FormData(); data.append('file', options.file as Blob)
      try {
        const response = await fetch('/api/uploads', { method: 'POST', body: data })
        const result = await response.json(); if (!response.ok) throw new Error(result.detail)
        kind === 'media' ? setMediaPath(result.path) : setRefPath(result.path)
        options.onSuccess?.(result); message.success(`${result.name} 已就绪`)
      } catch (error) { options.onError?.(error as Error) }
    },
  })

  const recognize = async () => {
    if (!mediaPath) return message.warning('请先上传视频或音频素材')
    const values = form.getFieldsValue()
    if (values.translationProvider === 'OpenAI 兼容接口' && (!values.translationUrl || !values.translationKey || !values.translationModel)) {
      return message.warning('请完整填写第三方中转平台的 Base URL、API Key 和模型名称')
    }
    const result = await jsonFetch('/api/projects/recognize', { method: 'POST', body: JSON.stringify({
      media: mediaPath,
      reference: refPath || null,
      asr_model: values.asrModel,
      source_language: values.sourceLanguage,
      preserve_background: values.preserveBackground,
      target_language: values.targetLanguage,
      translation_style: values.translationStyle,
      translation_provider: values.translationProvider,
      translation_url: values.translationUrl || '',
      translation_key: values.translationKey || '',
      translation_model: values.translationModel || '',
    }) })
    pollTask(result.task_id, data => { setProjectId(data.project_id); setRows(data.rows.map(rowFromApi)); setStage(1) })
  }

  const translate = async () => {
    if (!projectId) return message.warning('请先完成原文识别')
    const values = form.getFieldsValue()
    if (values.translationProvider === 'OpenAI 兼容接口' && (!values.translationUrl || !values.translationKey || !values.translationModel)) {
      return message.warning('请完整填写第三方中转平台的 Base URL、API Key 和模型名称')
    }
    const result = await jsonFetch(`/api/projects/${projectId}/translate`, { method: 'POST', body: JSON.stringify({ rows: rows.map(rowToApi), target_language: values.targetLanguage, style: values.translationStyle, provider: values.translationProvider, base_url: values.translationUrl || '', api_key: values.translationKey || '', model: values.translationModel || '' }) })
    pollTask(result.task_id, data => { setRows(data.rows.map(rowFromApi)); setStage(2) })
  }

  const synthesize = async () => {
    if (!projectId) return message.warning('请先创建项目')
    const values = form.getFieldsValue()
    const result = await jsonFetch(`/api/projects/${projectId}/synthesize`, { method: 'POST', body: JSON.stringify({ rows: rows.map(rowToApi), emotion: values.emotion, speed: values.speed, background_volume: values.backgroundVolume, burn_subtitles: values.burnSubtitles }) })
    pollTask(result.task_id, () => setStage(3))
  }

  const openProject = async () => {
    if (!projectId.trim()) return message.warning('请输入项目 ID')
    try { const data = await jsonFetch(`/api/projects/${projectId.trim()}`); setRows(data.rows.map(rowFromApi)); setStage(data.rows.some((r: any[]) => r[5]) ? 2 : 1); message.success('项目已打开') } catch (error: any) { message.error(error.message) }
  }

  const cancel = async () => { const data = await jsonFetch('/api/tasks/cancel', { method: 'POST' }); setTaskText(data.message) }

  const studioTabs = [
    { key: '0', label: '1 识别与首次翻译', children: <RecognizePane form={form} uploadProps={uploadProps} recognize={recognize} cancel={cancel} busy={busy} taskText={taskText} progress={progress} /> },
    { key: '1', label: '2 字幕审核与重新翻译', children: <ReviewPane form={form} rows={rows} columns={columns} translate={translate} busy={busy} /> },
    { key: '2', label: '3 音频克隆与合成', children: <SynthesizePane form={form} synthesize={synthesize} busy={busy} /> },
    { key: '3', label: '4 导出结果', children: <ResultPane projectId={projectId} /> },
  ]

  return <Layout className="app-shell">
    <Sider width={260} className="sidebar">
      <div className="logo"><div className="logo-mark">A</div><div><strong>AutoVid <i>Pro</i></strong><span>视频本地化与多语言制作平台</span></div></div>
      <Menu theme="dark" mode="inline" selectedKeys={[nav]} onClick={e => setNav(e.key)} items={[
        { key: 'studio', icon: <HomeOutlined />, label: '项目工作台' }, { key: 'voices', icon: <AudioOutlined />, label: '音色库' },
        { key: 'batch', icon: <InboxOutlined />, label: '批量任务' }, { key: 'models', icon: <DatabaseOutlined />, label: '模型管理' },
        { key: 'history', icon: <FileTextOutlined />, label: '任务记录' },
      ]} />
      <div className="sidebar-settings"><SettingOutlined /> 系统设置</div>
    </Sider>
    <Layout>
      <Header className="topbar"><span>让全球内容，触达更多人</span><Space size={20}><Text>● CUDA 正常</Text><Text>任务队列 1</Text><BellOutlined /><Avatar>U</Avatar><Text>User</Text></Space></Header>
      <Content className="content">
        {nav === 'studio' && <>
          <div className="page-head"><div><Title level={2}><TranslationOutlined /> 项目工作台</Title><Text type="secondary">视频翻译、字幕审核、配音合成，一站式完成</Text></div><Space><Input value={projectId} onChange={e => setProjectId(e.target.value)} placeholder="请选择项目或输入项目 ID" className="project-input" /><Button icon={<FolderOpenOutlined />} onClick={openProject}>打开已有项目</Button></Space></div>
          <Card className="steps-card"><Steps current={stage} items={[{ title: '识别与首次翻译', description: '识别语音并生成目标字幕' }, { title: '字幕审核与重新翻译', description: '校对原文与译文' }, { title: '音频克隆与合成', description: '确认字幕后直接配音' }, { title: '导出结果', description: '导出视频、字幕及音频' }]} /></Card>
          <Card className="workspace-card"><Form form={form} layout="vertical" initialValues={{ asrModel: 'large-v3-turbo', sourceLanguage: '中文', targetLanguage: '英语', translationStyle: '自然口语', translationProvider: '本地模型', preserveBackground: true, emotion: '自然', speed: 1, backgroundVolume: .75 }}><Tabs activeKey={String(stage)} onChange={key => setStage(Number(key))} items={studioTabs} /></Form></Card>
        </>}
        {nav === 'models' && <DataPage title="模型管理" icon={<RobotOutlined />} columns={['模型', '状态', '路径']} data={models} />}
        {nav === 'history' && <DataPage title="任务记录" icon={<FileTextOutlined />} columns={['时间', '文件名', '语言', '状态', '输出位置']} data={history} />}
        {nav === 'voices' && <EmptyPage title="音色库" icon={<AudioOutlined />} text="管理不同角色使用的声音参考与克隆音色" />}
        {nav === 'batch' && <EmptyPage title="批量任务" icon={<TeamOutlined />} text="一次导入多个视频并使用统一的翻译和配音设置" />}
      </Content>
    </Layout>
  </Layout>
}

function RecognizePane({ uploadProps, recognize, cancel, busy, taskText, progress }: any) {
  return <div>
    <div className="section-title"><CloudUploadOutlined /> 上传素材 <span>识别原始语音并立即生成首版目标字幕</span></div>
    <Row gutter={18}>
      <Col span={12}><Upload.Dragger {...uploadProps('media')} accept="video/*,audio/*"><PlayCircleFilled className="upload-icon blue" /><h3>视频或音频素材</h3><p>支持 MP4、MOV、MKV、AVI、MP3、WAV 等格式</p><Button type="primary" icon={<CloudUploadOutlined />}>选择文件</Button></Upload.Dragger></Col>
      <Col span={12}><Upload.Dragger {...uploadProps('reference')} accept="audio/*"><AudioOutlined className="upload-icon purple" /><h3>声音参考（可选）</h3><p>上传 4-8 秒清晰人声，用于后续声音克隆</p><Button icon={<CloudUploadOutlined />}>选择文件</Button></Upload.Dragger></Col>
    </Row>
    <div className="settings-grid four"><Form.Item name="asrModel" label="识别模型"><Select options={['large-v3-turbo', 'small'].map(value => ({ value }))} /></Form.Item><Form.Item name="sourceLanguage" label="原语言"><Select options={languages.map(value => ({ value }))} /></Form.Item><Form.Item name="targetLanguage" label="目标语言"><Select options={languages.map(value => ({ value }))} /></Form.Item><Form.Item name="translationStyle" label="翻译风格"><Select options={['自然口语','简洁解说','正式表达','忠实原文'].map(value => ({ value }))} /></Form.Item></div>
    <div className="translation-row"><Form.Item name="translationProvider" label="首次翻译后端"><Radio.Group options={[{ label: '本地模型', value: '本地模型' }, { label: '第三方中转平台（OpenAI 兼容）', value: 'OpenAI 兼容接口' }]} /></Form.Item><Form.Item name="preserveBackground" valuePropName="checked" label="音轨处理"><Checkbox>分离并保留背景音</Checkbox></Form.Item></div>
    <Form.Item noStyle shouldUpdate={(previous, current) => previous.translationProvider !== current.translationProvider}>{({ getFieldValue }) => getFieldValue('translationProvider') === 'OpenAI 兼容接口' && <ThirdPartyTranslationFields />}</Form.Item>
    <Space.Compact block><Button type="primary" size="large" icon={<PlayCircleFilled />} onClick={recognize} loading={busy} className="main-action">开始识别与首次翻译</Button><Button size="large" icon={<StopFilled />} onClick={cancel} className="stop-button">停止任务</Button></Space.Compact>
    {(busy || taskText !== '等待任务') && <div className="task-status"><Text strong>{taskText}</Text><Progress percent={progress} status={busy ? 'active' : 'normal'} /></div>}
  </div>
}

function ReviewPane({ rows, columns, translate, busy }: any) { return <div><div className="section-title"><TranslationOutlined /> 审核原文与目标字幕 <span>首版译文已生成；可以逐句修改，或更换设置后重新翻译</span></div><div className="settings-grid"><Form.Item name="targetLanguage" label="目标语言"><Select options={languages.map(value => ({ value }))} /></Form.Item><Form.Item name="translationStyle" label="翻译风格"><Select options={['自然口语','简洁解说','正式表达','忠实原文'].map(value => ({ value }))} /></Form.Item><Form.Item name="translationProvider" label="重新翻译后端"><Radio.Group options={[{ label: '本地模型', value: '本地模型' }, { label: '第三方中转平台（OpenAI 兼容）', value: 'OpenAI 兼容接口' }]} /></Form.Item></div><Form.Item noStyle shouldUpdate={(previous, current) => previous.translationProvider !== current.translationProvider}>{({ getFieldValue }) => getFieldValue('translationProvider') === 'OpenAI 兼容接口' && <ThirdPartyTranslationFields />}</Form.Item><Table columns={columns} dataSource={rows} scroll={{ x: 1200, y: 430 }} pagination={false} size="middle" bordered /><Button type="primary" size="large" block icon={<TranslationOutlined />} onClick={translate} loading={busy} className="bottom-action">重新翻译全部目标字幕</Button></div> }

function ThirdPartyTranslationFields() {
  const form = Form.useFormInstance()
  const { message } = AntApp.useApp()
  const [models, setModels] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const loadModels = async () => {
    const baseUrl = form.getFieldValue('translationUrl')
    const apiKey = form.getFieldValue('translationKey')
    if (!baseUrl || !apiKey) return message.warning('请先填写 Base URL 和 API Key')
    setLoading(true)
    try {
      const result = await jsonFetch('/api/remote-models', { method: 'POST', body: JSON.stringify({ base_url: baseUrl, api_key: apiKey }) })
      setModels(result.models)
      if (!result.models.includes(form.getFieldValue('translationModel'))) form.setFieldValue('translationModel', result.models[0])
      message.success(`已拉取 ${result.models.length} 个模型`)
    } catch (error: any) { message.error(error.message) } finally { setLoading(false) }
  }
  return <div className="api-settings">
    <Form.Item name="translationUrl" label="Base URL" required><Input placeholder="例如：https://api.example.com/v1" onChange={() => { setModels([]); form.setFieldValue('translationModel', undefined) }} /></Form.Item>
    <Form.Item name="translationKey" label="API Key" required><Input.Password placeholder="请输入中转平台密钥" onChange={() => { setModels([]); form.setFieldValue('translationModel', undefined) }} /></Form.Item>
    <Form.Item label="平台模型" required><Space.Compact block><Form.Item name="translationModel" noStyle><Select showSearch placeholder="请先拉取模型" options={models.map(value => ({ value }))} disabled={!models.length} /></Form.Item><Button onClick={loadModels} loading={loading}>拉取模型</Button></Space.Compact></Form.Item>
  </div>
}

function SynthesizePane({ synthesize, busy }: any) { return <div><div className="section-title"><AudioOutlined /> 克隆与合成 <span>目标字幕审核完成后再生成配音</span></div><div className="settings-grid four"><Form.Item name="emotion" label="表达风格"><Select options={['自然','平静','开心','严肃','激动','温柔','悲伤'].map(value => ({ value }))} /></Form.Item><Form.Item name="speed" label="语速"><Select options={[.8,1,1.1,1.2].map(value => ({ value, label: `${value}x` }))} /></Form.Item><Form.Item name="backgroundVolume" label="背景音量"><Select options={[.5,.65,.75,.9,1].map(value => ({ value }))} /></Form.Item><Form.Item name="burnSubtitles" valuePropName="checked" label="字幕"><Checkbox>烧录到视频</Checkbox></Form.Item></div><Button type="primary" size="large" block icon={<SaveOutlined />} onClick={synthesize} loading={busy}>审核完成，开始音频克隆</Button></div> }

function ResultPane({ projectId }: { projectId: string }) { return <div className="result-empty"><SaveOutlined /><Title level={4}>等待生成结果</Title><Text type="secondary">项目 {projectId || '尚未创建'} 完成合成后，可在这里下载视频、音频和字幕。</Text></div> }
function DataPage({ title, icon, columns, data }: any) { return <Card><Title level={3}>{icon} {title}</Title><Table columns={columns.map((title: string, i: number) => ({ title, dataIndex: i }))} dataSource={data.map((row: any[], key: number) => ({ key, ...row }))} /></Card> }
function EmptyPage({ title, icon, text }: any) { return <Card className="empty-page"><Title level={3}>{icon} {title}</Title><Text type="secondary">{text}</Text></Card> }

export default function Root() { return <AntApp><App /></AntApp> }
