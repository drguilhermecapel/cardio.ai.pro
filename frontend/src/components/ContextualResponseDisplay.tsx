import React from 'react'
import {
  Box,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Card,
  CardContent,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Divider,
} from '@mui/material'
import {
  ExpandMore,
  Info,
  Lightbulb,
  School,
  CameraAlt,
  Help,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
} from '@mui/icons-material'

interface ContextualResponse {
  message: string
  explanation?: string
  tips?: string[]
  visual_guide?: string
  educational_content?: {
    title?: string
    description?: string
    examples?: string[]
    key_features?: string[]
  }
  helpful_actions?: string[]
  humor_response?: string
  adaptive_suggestions?: string[]
}

interface ContextualResponseDisplayProps {
  response: ContextualResponse
  category: string
  confidence: number
  onTryAgain?: () => void
  onLearnMore?: () => void
  onGetHelp?: () => void
}

const ContextualResponseDisplay: React.FC<ContextualResponseDisplayProps> = ({
  response,
  category,
  confidence,
  onTryAgain,
  onLearnMore,
  onGetHelp,
}) => {
  const getCategoryIcon = (category: string) => {
    switch (category.toLowerCase()) {
      case 'medical_document':
        return <Info color="info" />
      case 'x_ray':
      case 'mri':
      case 'ct_scan':
        return <Warning color="warning" />
      case 'prescription':
        return <CheckCircle color="success" />
      case 'food':
      case 'animal':
      case 'person':
        return <ErrorIcon color="error" />
      default:
        return <Help color="action" />
    }
  }

  const getCategoryColor = (category: string): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (category.toLowerCase()) {
      case 'medical_document':
        return 'info'
      case 'x_ray':
      case 'mri':
      case 'ct_scan':
        return 'warning'
      case 'prescription':
        return 'success'
      case 'food':
      case 'animal':
      case 'person':
        return 'error'
      default:
        return 'default'
    }
  }

  return (
    <Box sx={{ width: '100%' }}>
      {/* Main Message */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        {getCategoryIcon(category)}
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6" gutterBottom>
            {response.message}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              label={`${category.replace('_', ' ')} detected`}
              color={getCategoryColor(category)}
              size="small"
            />
            <Chip
              label={`${(confidence * 100).toFixed(1)}% confidence`}
              variant="outlined"
              size="small"
            />
          </Box>
        </Box>
      </Box>

      {/* Humor Response */}
      {response.humor_response && (
        <Alert severity="info" sx={{ mb: 2 }}>
          <Typography variant="body2">
            {response.humor_response}
          </Typography>
        </Alert>
      )}

      {/* Explanation */}
      {response.explanation && (
        <Typography variant="body1" sx={{ mb: 2, color: 'text.secondary' }}>
          {response.explanation}
        </Typography>
      )}

      {/* Tips Section */}
      {response.tips && response.tips.length > 0 && (
        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <Lightbulb color="warning" />
              <Typography variant="h6">
                💡 Helpful Tips
              </Typography>
            </Box>
            <List dense>
              {response.tips.map((tip, index) => (
                <ListItem key={index} sx={{ py: 0.5 }}>
                  <ListItemText 
                    primary={`• ${tip}`}
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      )}

      {/* Visual Guide */}
      {response.visual_guide && (
        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <CheckCircle color="success" />
              <Typography variant="h6">
                📋 What an ECG looks like
              </Typography>
            </Box>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              {response.visual_guide}
            </Typography>
          </CardContent>
        </Card>
      )}

      {/* Educational Content */}
      {response.educational_content && (
        <Accordion sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <School color="primary" />
              <Typography variant="h6">
                📚 {response.educational_content.title || 'Learn More About ECGs'}
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            {response.educational_content.description && (
              <Typography variant="body2" sx={{ mb: 2 }}>
                {response.educational_content.description}
              </Typography>
            )}
            
            {response.educational_content.key_features && response.educational_content.key_features.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Key ECG Features:
                </Typography>
                <List dense>
                  {response.educational_content.key_features.map((feature, index) => (
                    <ListItem key={index} sx={{ py: 0 }}>
                      <ListItemIcon sx={{ minWidth: 24 }}>
                        <CheckCircle color="success" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText 
                        primary={feature}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}

            {response.educational_content.examples && response.educational_content.examples.length > 0 && (
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Examples:
                </Typography>
                <List dense>
                  {response.educational_content.examples.map((example, index) => (
                    <ListItem key={index} sx={{ py: 0 }}>
                      <ListItemText 
                        primary={`• ${example}`}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </AccordionDetails>
        </Accordion>
      )}

      {/* Adaptive Suggestions */}
      {response.adaptive_suggestions && response.adaptive_suggestions.length > 0 && (
        <Card sx={{ mb: 2, bgcolor: 'primary.50' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              🎯 Personalized Suggestions
            </Typography>
            <List dense>
              {response.adaptive_suggestions.map((suggestion, index) => (
                <ListItem key={index} sx={{ py: 0.5 }}>
                  <ListItemText 
                    primary={`• ${suggestion}`}
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      )}

      {/* Helpful Actions */}
      {response.helpful_actions && response.helpful_actions.length > 0 && (
        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              🚀 What you can do next
            </Typography>
            <List dense>
              {response.helpful_actions.map((action, index) => (
                <ListItem key={index} sx={{ py: 0.5 }}>
                  <ListItemText 
                    primary={`• ${action}`}
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      )}

      <Divider sx={{ my: 2 }} />

      {/* Action Buttons */}
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'center' }}>
        <Button 
          variant="contained" 
          startIcon={<CameraAlt />}
          onClick={onTryAgain}
          sx={{ minWidth: 120 }}
        >
          📷 Try Again
        </Button>
        
        {response.educational_content && (
          <Button 
            variant="outlined" 
            startIcon={<School />}
            onClick={onLearnMore}
            sx={{ minWidth: 120 }}
          >
            📚 Learn More
          </Button>
        )}
        
        <Button 
          variant="outlined" 
          startIcon={<Help />}
          onClick={onGetHelp}
          sx={{ minWidth: 120 }}
        >
          💬 Get Help
        </Button>
      </Box>

      {/* Privacy Notice */}
      <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          🔒 Privacy Notice: Your image was not stored since it doesn't contain ECG data.
        </Typography>
      </Box>
    </Box>
  )
}

export default ContextualResponseDisplay
