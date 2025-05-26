import React from 'react';
import { 
  Container, 
  Heading, 
  Box, 
  Divider 
} from '@chakra-ui/react';
import ModelSelector from '../components/ModelSelector';

const Settings = () => {
  return (
    <Container maxW="container.lg" py={8}>
      <Heading mb={6}>Settings</Heading>
      
      <Box mb={8}>
        <ModelSelector />
      </Box>
      
      <Divider my={6} />
      
      {/* Additional settings can be added here */}
    </Container>
  );
};

export default Settings; 