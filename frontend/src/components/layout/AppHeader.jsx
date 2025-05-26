import React from 'react';
import { Layout, Button, Typography, Space, Switch } from 'antd';
import { BulbOutlined, UserOutlined } from '@ant-design/icons';

const { Header } = Layout;
const { Title } = Typography;

const AppHeader = ({ toggleDarkMode, isDarkMode }) => {
  return (
    <Header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 16px' }}>
      <Title level={4} style={{ margin: 0, color: 'white' }}>
        DSAgency
      </Title>
      
      <Space>
        <Switch
          checkedChildren={<BulbOutlined />}
          unCheckedChildren={<BulbOutlined />}
          checked={isDarkMode}
          onChange={toggleDarkMode}
        />
        <Button type="text" icon={<UserOutlined />} style={{ color: 'white' }}>
          Profile
        </Button>
      </Space>
    </Header>
  );
};

export default AppHeader; 