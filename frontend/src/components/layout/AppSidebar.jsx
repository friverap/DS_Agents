import React, { useState } from 'react';
import { Layout, Menu } from 'antd';
import { Link, useLocation } from 'react-router-dom';
import {
  DashboardOutlined,
  AppstoreOutlined,
  SettingOutlined,
  CloudUploadOutlined,
  ExperimentOutlined,
  FundOutlined,
  CodeOutlined,
  RobotOutlined,
  SafetyOutlined,
  CloudServerOutlined,
  MonitorOutlined
} from '@ant-design/icons';

const { Sider } = Layout;

const AppSidebar = () => {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();
  
  const items = [
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: <Link to="/dashboard">Dashboard</Link>,
    },
    {
      key: 'workflows',
      icon: <AppstoreOutlined />,
      label: <Link to="/workflows">Workflows</Link>,
    },
    {
      key: 'ds-workflow',
      icon: <ExperimentOutlined />,
      label: 'DS Workflow',
      children: [
        {
          key: 'data-collection',
          icon: <CloudUploadOutlined />,
          label: <Link to="/data-collection">Data Collection</Link>,
        },
        {
          key: 'preprocessing',
          icon: <CodeOutlined />,
          label: <Link to="/preprocessing">Preprocessing</Link>,
        },
        {
          key: 'exploratory-analysis',
          icon: <FundOutlined />,
          label: <Link to="/exploratory-analysis">Exploratory Analysis</Link>,
        },
        {
          key: 'feature-engineering',
          icon: <RobotOutlined />,
          label: <Link to="/feature-engineering">Feature Engineering</Link>,
        },
        {
          key: 'model-training',
          icon: <ExperimentOutlined />,
          label: <Link to="/model-training">Model Training</Link>,
        },
        {
          key: 'validation',
          icon: <SafetyOutlined />,
          label: <Link to="/validation">Validation</Link>,
        },
        {
          key: 'deployment',
          icon: <CloudServerOutlined />,
          label: <Link to="/deployment">Deployment</Link>,
        },
        {
          key: 'monitoring',
          icon: <MonitorOutlined />,
          label: <Link to="/monitoring">Monitoring</Link>,
        },
      ],
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: <Link to="/settings">Settings</Link>,
    },
  ];
  
  return (
    <Sider collapsible collapsed={collapsed} onCollapse={(value) => setCollapsed(value)}>
      <div style={{ height: '32px', margin: '16px', background: 'rgba(255, 255, 255, 0.2)' }} />
      <Menu
        theme="dark"
        defaultSelectedKeys={['dashboard']}
        selectedKeys={[location.pathname.split('/')[1] || 'dashboard']}
        mode="inline"
        items={items}
      />
    </Sider>
  );
};

export default AppSidebar; 