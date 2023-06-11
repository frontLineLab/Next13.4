import React from 'react'

interface Props {
  children: React.ReactNode
}

export default function DashboardLayout({
    children,
  }: Props) {
    return (
      <section>
        <div>DashboardLayout</div>
        {children}
      </section>
    )
  }