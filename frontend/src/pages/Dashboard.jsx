  import React, { useState, useEffect } from 'react';
import { Layout, Typography, Card, Row, Col, Statistic, Button, Divider } from 'antd';
import { ProjectOutlined, CheckCircleOutlined, ExperimentOutlined, RocketOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import ChatInterface from '../components/ChatInterface';
import { modelsApi } from '../services/api';

const { Content } = Layout;
const { Title } = Typography;

const Dashboard = () => {
  const [currentModel, setCurrentModel] = useState({
    provider: '',
    model: ''
  });

  // Fetch current model configuration
  useEffect(() => {
    const fetchCurrentModel = async () => {
      try {
        const response = await modelsApi.getCurrentModel();
        setCurrentModel({
          provider: response.data.provider,
          model: response.data.model
        });
      } catch (error) {
        console.error('Error fetching current model:', error);
      }
    };

    fetchCurrentModel();
  }, []);

  return (
    <Content style={{ margin: '16px' }}>
      <div style={{ padding: 24, background: '#fff', minHeight: 360 }}>
        <Title level={2}>Dashboard</Title>
        
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Card>
              <Statistic 
                title="Total Workflows" 
                value={8} 
                prefix={<ProjectOutlined />} 
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic 
                title="Completed" 
                value={5} 
                prefix={<CheckCircleOutlined />} 
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic 
                title="Active Experiments" 
                value={3} 
                prefix={<ExperimentOutlined />} 
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic 
                title="Deployed Models" 
                value={2} 
                prefix={<RocketOutlined />} 
                valueStyle={{ color: '#722ed1' }}
              />
            </Card>
          </Col>
        </Row>
        
        <div style={{ marginTop: 32 }}>
          <Title level={4}>Quick Actions</Title>
          <Row gutter={[16, 16]}>
            <Col>
              <Button type="primary">
                <Link to="/workflows/create">New Workflow</Link>
              </Button>
            </Col>
            <Col>
              <Button>
                <Link to="/data-collection">Import Data</Link>
              </Button>
            </Col>
            <Col>
              <Button>
                <Link to="/model-training">Train Model</Link>
              </Button>
            </Col>
          </Row>
        </div>

        <Divider />
        
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <ChatInterface 
              modelProvider={currentModel.provider} 
              modelName={currentModel.model}
            />
          </Col>
        </Row>
      </div>
    </Content>
  );
};

export default Dashboard; 