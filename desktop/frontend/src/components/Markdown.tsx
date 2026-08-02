import type { ReactNode } from "react"

function rich(text: string): ReactNode[] {
  const parts = text.split(/\*\*(.+?)\*\*/g)
  return parts.map((p, i) => (i % 2 === 1 ? <strong key={i}>{p}</strong> : p))
}

export default function Markdown({ text }: { text: string }): React.JSX.Element {
  const lines = text.split("\n")
  const out: ReactNode[] = []
  let list: string[] = []

  const flush = (key: number): void => {
    if (list.length > 0) {
      out.push(
        <ul className="md-list" key={key}>
          {list.map((li, i) => (
            <li key={i}>{rich(li)}</li>
          ))}
        </ul>
      )
      list = []
    }
  }

  lines.forEach((raw, i) => {
    const line = raw.trim()
    if (!line) {
      flush(i)
      return
    }
    if (line === "---") {
      flush(i)
      out.push(<hr key={`hr-${i}`} className="md-hr" />)
      return
    }
    if (line.startsWith("## ")) {
      flush(i)
      out.push(<h2 className="md-h2" key={`h2-${i}`}>{line.slice(3)}</h2>)
      return
    }
    if (line.startsWith("# ")) {
      flush(i)
      out.push(<h1 className="md-h1" key={`h1-${i}`}>{line.slice(2)}</h1>)
      return
    }
    if (line.startsWith("* ") || line.startsWith("- ")) {
      list.push(line.slice(2))
      return
    }
    flush(i)
    out.push(<p className="md-p" key={`p-${i}`}>{rich(line)}</p>)
  })
  flush(lines.length)

  return <div className="markdown">{out}</div>
}
